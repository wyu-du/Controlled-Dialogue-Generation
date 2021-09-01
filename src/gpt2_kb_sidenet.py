# -*- coding: utf-8 -*-

import json
import argparse
import time
import os
from datetime import datetime
from utils import parse_profile_data, batchify_kb_data, prepare_eval_kb, test_ppl_gpt2_kb, make_batch_kb_data
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

# Device and fp16 management.
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from packaging import version
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast



class Seq2SeqModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_dim = config.n_embd
        self.transformer = GPT2Model.from_pretrained("microsoft/DialoGPT-medium")
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.side_init()
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
    def side_init(self):
        self.rnn = nn.LSTM(self.model_dim, self.model_dim, 
                           num_layers=1, bidirectional=True, batch_first=True)
        self.W_p = nn.Linear(2*self.model_dim, 2*self.model_dim, bias=False)
        self.W_c = nn.Linear(1, 2*self.model_dim, bias=False)
        self.W_d = nn.Linear(self.model_dim, 2*self.model_dim)
        self.v = nn.Linear(2*self.model_dim, 1, bias=False)
        self.alpha_linear = nn.Linear(4*self.model_dim, 1)
        self.mlp = nn.Linear(3*self.model_dim, self.model_dim)
        self.fuse = nn.Linear(2*self.model_dim, 1)
        
    def get_padding_mask(self, profile):
        self.pad_id = self.transformer.wte.weight.size(0) 
        ones = torch.ones_like(profile)
        zeros = torch.zeros_like(profile)
        pad_mask = torch.where(profile==self.pad_id, zeros, ones)
        seq_lens = pad_mask.sum(dim=1)
        profile_revised = torch.where(profile==self.pad_id, ones*self.config.eos_token_id, profile)
        return pad_mask, seq_lens, profile_revised
    
    def get_input_mask(self, input_ids):
        self.pad_id = self.transformer.wte.weight.size(0) 
        ones = torch.ones_like(input_ids)
        input_ids = torch.where(input_ids==self.pad_id, ones*self.config.eos_token_id, input_ids)
        return input_ids
        
    def side_net(self, profile, base_output, coverage):
        """
        Compute cross attention between profile and base_output.
            - profile: B x L
            - base_output: B x S x D
            - coverage: B x L
        """
        enc_padding_mask, seq_lens, profile = self.get_padding_mask(profile)
        kb_emb = self.transformer.wte(profile) # B x L x D
        enc_padding_mask = enc_padding_mask.type_as(kb_emb)
        
        packed = pack_padded_sequence(kb_emb, seq_lens, 
                                      batch_first=True, 
                                      enforce_sorted=False)
        self.rnn.flatten_parameters()
        output, _ = self.rnn(packed)
        kb_outputs, _ = pad_packed_sequence(output, 
                                            batch_first=True, 
                                            total_length=profile.size(1)) # B x L x 2D
        profile_fea = self.W_p(kb_outputs.view(-1, 2*self.model_dim)) # BL x 2D
        
        bz, seq_len, d = kb_outputs.size()
        final_list = []
        coverage_loss = 0.
        for tid in range(base_output.size(1)):
            base_fea = self.W_d(base_output[:, tid, :]) # B x 2D
            base_fea_ext = base_fea.unsqueeze(1).expand(bz, seq_len, d).contiguous()
            base_fea_ext = base_fea_ext.view(-1, d) # BL x 2D
            
            coverage_fea = self.W_c(coverage.contiguous().view(-1, 1)) # BL x 2D
            
            att_fea = profile_fea + base_fea_ext + coverage_fea
            e = torch.tanh(att_fea) # BL x 2D
            scores = self.v(e) # BL x 1
            scores = scores.view(-1, seq_len) # B x L
            
            attn_dist_ = F.softmax(scores, dim=1) * enc_padding_mask # B x L
            normalization_factor = attn_dist_.sum(1, keepdim=True)
            attn_dist = attn_dist_ / normalization_factor # B x L
            
            ctx_t = torch.bmm(attn_dist.unsqueeze(1), kb_outputs) # B x 1 x 2D
            ctx_t = ctx_t.squeeze(1) # B x 2D
            ctx_t = torch.tanh(ctx_t)
            
            coverage = coverage + attn_dist # B x L
            
            h_side = self.mlp(torch.cat((ctx_t, base_output[:, tid, :]), dim=-1)) # B x D
            h_side = torch.tanh(h_side)
            alpha2 = self.fuse(torch.cat((h_side, base_output[:, tid, :]), dim=-1))
            alpha2 = torch.sigmoid(alpha2)
            h_t = (1-alpha2) * h_side + alpha2 * base_output[:, tid, :]
            logits = self.lm_head(h_t)
            vocab_dist = F.softmax(logits, dim=-1)
            
            alpha_input = torch.cat((ctx_t, base_fea), dim=-1) # B x 4D
            alpha = self.alpha_linear(alpha_input)
            alpha = torch.sigmoid(alpha)
            vocab_dist_ = alpha * vocab_dist   # B x V
            attn_dist_ = (1-alpha) * attn_dist # B x L
            final_dist = vocab_dist_.scatter_add(1, profile, attn_dist_) # B x V
            final_list.append(final_dist)
            
            coverage_loss += torch.sum(torch.min(attn_dist, coverage), 1) # B
        
        lm_probs = torch.stack(final_list, dim=1) # B x S x V
        lm_logits = torch.log(lm_probs + 1e-12) # B x S x V
        return lm_logits, coverage_loss.mean(), coverage
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past=None, 
        **kwargs
    ):
        token_type_ids = kwargs.get("token_type_ids", None)
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None
        return {
            "input_ids": input_ids,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "decoder_profile": kwargs.get('decoder_profile'),
            "decoder_coverage": kwargs['decoder_coverage']
        }
        
    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_profile=None,
        decoder_coverage=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        input_ids = self.get_input_mask(input_ids)

        # Extract features from base model
        with torch.no_grad():
            transformer_outputs = self.transformer(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        hidden_states = transformer_outputs[0]
        
        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        base_output = hidden_states
        if decoder_coverage is None:
            decoder_coverage = torch.zeros_like(decoder_profile)
        decoder_coverage = decoder_coverage.type_as(base_output)
        
        # get side outputs
        lm_logits, coverage_loss, decoder_coverage = self.side_net(decoder_profile, base_output, decoder_coverage) # B x S x V
        
        # Freeze the base model
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.NLLLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), 
                            shift_labels.view(-1))
            loss = loss + self.lamb*coverage_loss

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )



class Seq2SeqTrainer(Trainer):
  """Class to finetune a Seq2Seq model."""
  def __init__(
    self,
    num_beams=4, 
    max_length=32,
    *args, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.num_beams = num_beams
    self.max_length = max_length
    
  def compute_loss(self, model, inputs):
    """
    How the loss is computed by Trainer. By default, all models return the loss in the first element.

    Subclass and override for custom behavior.
    """
    outputs = model(input_ids=inputs['input_ids'],
                    labels=inputs['input_ids'],
                    decoder_profile=inputs['decoder_profile'])
    # Save past state if it exists
    if self.args.past_index >= 0:
        self._past = outputs[self.args.past_index]

    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    return outputs["loss"] if isinstance(outputs, dict) else outputs[0]

  def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
    """
    Runs the model to either generate a sequence and/or compute the loss.
    """
    has_labels = all(inputs.get(k) is not None for k in self.label_names)
    inputs = self._prepare_inputs(inputs)
    # Compute loss with labels first.
    with torch.no_grad():
      if self.args.fp16 and _use_native_amp:
        with autocast():
          outputs = model(input_ids=inputs['input_ids'],
                          labels=inputs['input_ids'],
                          decoder_profile=inputs['decoder_profile'])
      else:
        outputs = model(input_ids=inputs['input_ids'],
                        labels=inputs['input_ids'],
                        decoder_profile=inputs['decoder_profile'])
      if has_labels:
        loss = outputs[0].mean().detach()
      else:
        loss = None
    # If we're only computing the conditional log-likelihood, return.
    if prediction_loss_only:
      return (loss, None, None)
    # Otherwise run model.generate() to get predictions.
    decoder_profile = inputs['decoder_profile'].repeat_interleave(self.num_beams,dim=0)
    model_kwargs = {
      "decoder_profile": decoder_profile,
      "decoder_coverage": torch.zeros_like(decoder_profile)
    }
    if isinstance(model, torch.nn.DataParallel):
      preds = model.module.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        num_beams=self.num_beams,
        max_length=self.max_length,
        **model_kwargs
      )
    else:
      preds = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        num_beams=self.num_beams,
        max_length=self.max_length,
        **model_kwargs
      )
    if len(preds) == 1:
      preds = preds[0]
    # Pad predictions if necessary so they can be concatenated across batches.
    if preds.shape[-1] < self.max_length:
      preds = torch.nn.functional.pad(
        preds, (0, self.max_length-preds.shape[-1]),
        mode='constant',
        value=self.tokenizer.pad_token_id
      )
    # Post-process labels.
    if has_labels:
      labels = inputs.get('input_ids')
    else:
      labels = None
    return (loss, preds, labels)



def train(args):
    # Load the dataset
    df = parse_profile_data(in_file=f'../data/{args.dataset}.json', mode='train')
    train_len = 153080
    trn_df = df.iloc[:train_len, :]
    val_df = df.iloc[train_len:, :]
    
    # Load the pre-trained model
    ckpt_path = None
    if args.task == 'train':
        ckpt_path = args.model_name
    else:
        ckpt_path = f"{args.model_name}_{args.dataset}_base_{args.timestamp}/checkpoint-{args.ckpt}"
        # update timestamp and create new path for ckpt
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    print(f"Vocab size: {len(tokenizer)}")
    
    train_data_tokenized = batchify_kb_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_kb_data(val_df, tokenizer, args)
    
    model = Seq2SeqModel.from_pretrained(ckpt_path)
    model.lamb = args.lamb
    model = model.to('cuda:0')
    
    # Training Setup
    train_args = TrainingArguments(
        output_dir=f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=1000,
        logging_steps=100, 
        # optimization args, the trainer uses the Adam optimizer
        # and has a linear warmup for the learning rate
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=1e-04,
        num_train_epochs=10,
        warmup_steps=0, 
        lr_scheduler_type='constant',
        # misc args
        seed=42,
        save_total_limit=1, # limit the total amount of checkpoints
        disable_tqdm=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        local_rank=args.local_rank
    )
    
    trainer = Seq2SeqTrainer(
        num_beams=args.beam_size, 
        max_length=args.decoder_max_length,
        model=model,
        args=train_args,
        train_dataset=train_data_tokenized,
        eval_dataset=valid_data_tokenized,
        tokenizer=tokenizer,
    )
    
    # Now that we have the trainer set up, we can finetune.
    trainer.train()



def generate_sentences(batch,
                       model, 
                       tokenizer,
                       args, 
                       device='cuda:0'):
  # Create batch inputs.
  features, kbs = make_batch_kb_data(batch, tokenizer, args, device)
  decoder_profile = kbs['input_ids']
  model_kwargs = {
      "decoder_profile": decoder_profile.repeat_interleave(args.beam_size,dim=0),
      "decoder_coverage": torch.zeros_like(decoder_profile)
  }
  # Generate with beam search.
  max_gen_len = args.max_generation_length + features['input_ids'].size(1)
  generated_ids = model.generate(
      input_ids=features['input_ids'], 
      attention_mask=features['attention_mask'],
      num_beams=args.beam_size,
      do_sample=True,
      top_k=args.top_k,
      top_p=args.top_p,
      max_length=max_gen_len,
      **model_kwargs
  )
  
  generated_sentences = []
  for i, gen_ids in enumerate(generated_ids):
      sid = int(torch.sum(features['attention_mask'][i]).data)
      gen_ids = gen_ids[sid:]
      gen_sent = tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True).strip()
      generated_sentences.append(gen_sent)
  return generated_sentences



def test(args):
    if args.dataset == 'convai2_raw':
        te_df = parse_profile_data(in_file=f'../data/{args.dataset}.json', mode='validation')
    else:
        te_df = parse_profile_data(in_file=f'../data/{args.dataset}.json', mode='test')
    print('Data loaded!!!')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model.lamb = args.lamb
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'generated': generate_sentences(
            batch,
            model, 
            tokenizer, 
            args,
            device='cuda:0')
        },
        batched=True, 
        batch_size=args.batch_size,
    )
    
    # Compute ppl
    ppl = test_ppl_gpt2_kb(te_df, model, tokenizer, args)
    ppl = round(ppl, 4)
    print(f"Test ppl: {ppl}")
      
    # prepare evaluation data
    ref_list, pred_list, da_list = prepare_eval_kb(list(test_output))
    reference_dict = {
        "language": "en",
        "values": ref_list,  
    }
    ref_da_dict = {
        "language": "en",
        "values": da_list,  
    }
    prediction_dict = {
        "language": "en",
        "test ppl": ppl,
        "values": pred_list,  
    }
    
    if args.timestamp == '0':
        os.makedirs(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}")
    
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs.json", 'w') as f:
      f.write(json.dumps(reference_dict, indent=2))
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs_profiles.json", 'w') as f:
      f.write(json.dumps(ref_da_dict, indent=2))
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json", 'w') as f:
      f.write(json.dumps(prediction_dict, indent=2))



def decode_time(args):
    te_df = parse_profile_data(in_file=f'../data/{args.dataset}.json', mode='validation')
    te_df = te_df.iloc[int(0.8*len(te_df)):,:]
    print('Data loaded!!!')
    
    # Load the model
    if args.timestamp == '0':
        tokenizer = GPT2Tokenizer.from_pretrained(f"{args.model_name}")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    if args.timestamp == '0':
        model = Seq2SeqModel.from_pretrained(f"{args.model_name}")
    else:
        model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    df = te_df.iloc[:10]
    a = datetime.now() 
    test_output = Dataset.from_pandas(df).map(
        lambda batch: {'generated': generate_sentences(
            batch,
            model, 
            tokenizer, 
            args,
            device='cuda:0')
        },
        batched=True, 
        batch_size=1,
    )
    b = datetime.now() 
    dec_time = (b-a).seconds
    
    total_words = 0
    for item in list(test_output):
        sent = item['generated']
        total_words += len(sent.split())
    
    avg_dec = dec_time/total_words
    print('Avg decoding time:', str(avg_dec))



if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default="train", 
                   help="specify the task to do: (train)ing, ft(finetune), (eval)uation")
    p.add_argument('-c', '--ckpt', type=str, default="47839", 
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-04-26-10-21-06', 
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', 
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="convai2_raw", 
                   help="specify the dataset: dailydialog_raw, convai2_raw")
    p.add_argument('--model_name', type=str, default="microsoft/DialoGPT-medium", 
                   help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('--lamb', type=float, default=1e-5)
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--encoder_max_length', type=int, default=200)
    p.add_argument('--decoder_max_length', type=int, default=80)
    p.add_argument('--max_generation_length', type=int, default=60)
    p.add_argument('--beam_size', type=int, default=1)
    p.add_argument('--top_k', type=int, default=10)
    p.add_argument('--top_p', type=float, default=1.0)
    p.add_argument('--do_sample', action='store_true', 
                   help="specify whether sampling from output distribution during generation")
    p.add_argument('--local_rank', type=int, default=-1, 
                   help="Multiple GPU training")
    args = p.parse_args()
    
    if args.task == 'train':
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        train(args)
    elif args.task == 'ft':
        train(args)
    elif args.task == 'eval':
        test(args)
    elif args.task == 'dect':
        decode_time(args)
    else:
        print('Unrecognizable task!!!')