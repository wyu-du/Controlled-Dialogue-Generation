# -*- coding: utf-8 -*-

import json
import argparse
import time
import os
from datetime import datetime
from utils import parse_data, batchify_data, prepare_eval, test_ppl_gpt2, make_batch_data
from datasets import Dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions
from transformers.modeling_outputs import SequenceClassifierOutput

# Device and fp16 management.
import torch
from torch import nn
from torch.nn import functional as F
from packaging import version
if version.parse(torch.__version__) < version.parse("1.6"):
    from transformers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast

# Special tokens
SPECIAL_TOKENS = {
"inform": "[inform]",
"question": "[question]",
"directive": "[directive]",
"commissive": "[commissive]",  
}



class Seq2SeqModel(GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.model_dim = config.n_embd
        self.transformer = GPT2Model.from_pretrained("microsoft/DialoGPT-medium")
        self.num_labels = len(SPECIAL_TOKENS.keys())
        self.classifier = nn.Linear(self.model_dim, self.num_labels)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.side_init()
        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None
        
    def side_init(self):
        self.side_emb = nn.Embedding(4, self.model_dim)
        self.intent_layer = nn.Linear(self.model_dim, self.model_dim)
        self.W1 = nn.Linear(self.model_dim*2, self.model_dim, bias=False)
        
    def get_input_mask(self, input_ids):
        self.pad_id = self.transformer.wte.weight.size(0) # 50257
        ones = torch.ones_like(input_ids)
        mask = torch.zeros_like(input_ids)
        input_ids = torch.where(input_ids==self.pad_id, ones*self.config.eos_token_id, input_ids) # 50256
        for i in range(input_ids.size(0)):
            ctx_len = 0
            for j in range(input_ids.size(1)):
                if input_ids[i,j]==self.config.eos_token_id:
                    ctx_len += 1
                if ctx_len==5 and input_ids[i,j]!=self.config.eos_token_id:
                    mask[i,j] += 1
        return input_ids, mask
        
    def side_net(self, intents, base_output):
        intents = intents.unsqueeze(1) # B x 1
        intent_emb = self.side_emb(intents) # B x 1 x E
        intent_emb = self.intent_layer(intent_emb) # B x 1 x D
        intent_emb = intent_emb.repeat(1, base_output.size(1), 1) #  B x S x D
        inp_emb = torch.cat([intent_emb, base_output], dim=2) # B x S x 2D
        h_1 = self.W1(inp_emb) # B x S x D
        h_3 = torch.tanh(h_1) # B x S x D
        return h_3
    
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
            "decoder_da_inputs": kwargs['decoder_da_inputs'],
            "decoder_dis_loss": False
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
        decoder_da_inputs=None,
        decoder_dis_loss=True,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            ``labels = input_ids`` Indices are selected in ``[-100, 0, ..., config.vocab_size]`` All labels set to
            ``-100`` are ignored (masked), the loss is only computed for labels in ``[0, ..., config.vocab_size]``
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        intents = decoder_da_inputs
        if intents[0] > self.config.vocab_size:
            intents -= self.config.vocab_size+1
        input_ids, input_mask = self.get_input_mask(input_ids)

        # Extract features from base model
        with torch.no_grad():
            transformer_outputs = self.transformer(
                input_ids,
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
        
        if not self.pretrain_clf:
            # get side outputs
            side_output = self.side_net(intents, base_output) # B x S x D
            # fuse base and side output
            sequence_output = base_output * torch.sigmoid(self.alpha) + side_output * (1 - torch.sigmoid(self.alpha))
            lm_logits = self.lm_head(sequence_output)
        
        # compute classification logits
        if decoder_dis_loss:
            if self.pretrain_clf:
                sequence_output = base_output
            else:
                sequence_output = side_output
            input_mask = input_mask.type_as(sequence_output)       # B x S
            seq_len = input_mask.sum(dim=1, keepdim=True)          # B x 1
            clf_output = sequence_output * input_mask.unsqueeze(2) # B x S x D
            clf_output = clf_output.sum(dim=1)                     # B x D
            clf_output = clf_output / (seq_len + 1e-10)            # B x D
            clf_logits = self.classifier(clf_output)               # B x 4
        
        # Freeze the base model
        for param in self.transformer.parameters():
            param.requires_grad = False
        # Freeze the classifier
        if not self.pretrain_clf:
            for param in self.classifier.parameters():
                param.requires_grad = False
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
#            labels, label_mask = self.get_input_mask(labels)
            if not self.pretrain_clf:
                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                # Flatten the tokens
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                pad_mask = attention_mask[..., 1:].contiguous().view(-1)
                pad_mask = pad_mask.type_as(loss)
                loss = loss * pad_mask
                loss = loss.mean()
            
            if decoder_dis_loss:
                dis_loss = loss_fct(clf_logits.view(-1, self.num_labels), intents.view(-1))
                seq_len = attention_mask[..., 1:].sum(dim=1)
                dis_loss = dis_loss * seq_len.type_as(dis_loss)
                dis_loss = dis_loss.mean()
                if self.pretrain_clf:
                    loss = dis_loss
                else:
#                    print('lm_loss', loss)
#                    print('dis_loss', dis_loss)
#                    print('scaled_lm_loss', self.lamb * loss)
#                    print('scaled_dis_loss', (1-self.lamb) * dis_loss)
                    loss = loss + self.lamb * dis_loss
#                    print('loss', loss)

        if self.pretrain_clf:
            if not return_dict:
                output = (clf_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=clf_logits,
                    hidden_states=transformer_outputs.hidden_states,
                    attentions=transformer_outputs.attentions,
                )
        else:
            if not return_dict:
                output = (lm_logits,) + transformer_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
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
                    attention_mask=inputs['attention_mask'],
                    decoder_da_inputs=inputs['labels'][:,0])
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
                          attention_mask=inputs['attention_mask'],
                          decoder_da_inputs=inputs['labels'][:,0])
      else:
        outputs = model(input_ids=inputs['input_ids'],
                        labels=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        decoder_da_inputs=inputs['labels'][:,0])
      if has_labels:
        loss = outputs[0].mean().detach()
      else:
        loss = None
    # If we're only computing the conditional log-likelihood, return.
    if prediction_loss_only:
      return (loss, None, None)
    # Otherwise run model.generate() to get predictions.
    model_kwargs = {
      "decoder_da_inputs": inputs['labels'][:,0].repeat_interleave(self.num_beams,dim=0)
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
    trn_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='train')
    val_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='validation')
    
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
    tokenizer.add_tokens([i for i in SPECIAL_TOKENS.values()])
    print(f"Vocab size: {len(tokenizer)}")
    
    train_data_tokenized = batchify_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_data(val_df, tokenizer, args)
    
    model = Seq2SeqModel.from_pretrained(ckpt_path)
    model.pretrain_clf = args.pretrain_clf
    model.lamb = args.lamb
    model = model.to('cuda:0')
    
    # Training Setup
    train_args = TrainingArguments(
        output_dir=f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}",
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=5000,
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
    
    
    
def generate_label(batch, model, tokenizer, args, device='cuda:0'):
  """
  Function to generate da label. 
  """
  # Create batch inputs.
  features, labels = make_batch_data(batch, tokenizer, args, device)
  da_ids = labels['input_ids'][:, 0]
  # Generate preds.
  with torch.no_grad():
      outputs = model(input_ids=features['input_ids'],
                  decoder_da_inputs=da_ids,
                  decoder_dis_loss=True)
  logits = F.softmax(outputs.logits, dim=-1)
  preds = torch.argmax(logits, dim=-1)
  correct = 0
  for i in range(preds.size(0)):
      if preds[i] == da_ids[i]:
          correct += 1
  return [correct]



def generate_sentences(batch,
                       model,
                       tokenizer,
                       args, 
                       device='cuda:0'):
  # Create batch inputs.
  features, labels = make_batch_data(batch, tokenizer, args, device)
  da_ids = labels['input_ids'][:, 0]
  # add encoder_outputs
  model_kwargs = {
      "decoder_da_inputs": da_ids.repeat_interleave(args.beam_size,dim=0),
      "decoder_dis_loss": False,
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
  print(generated_sentences[0])
  return generated_sentences



def test(args):
    te_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='test')
    print('Data loaded!!!')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model.pretrain_clf = args.pretrain_clf
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
    
    # Print aplha
    alpha = torch.sigmoid(model.alpha)
    alpha = round(alpha.data.item(), 4)
    print('Alpha:', alpha)
    
    # Compute ppl
    ppl = test_ppl_gpt2(te_df, model, tokenizer, args)
    ppl = round(ppl, 4)
    print(f"Test ppl: {ppl}")
      
    # prepare evaluation data
    ref_list, pred_list, da_list = prepare_eval(list(test_output))
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
        "alpha": alpha,
        "test ppl": ppl,
        "values": pred_list,  
    }
    
    if args.timestamp == '0':
        os.makedirs(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}")
    
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs.json", 'w') as f:
      f.write(json.dumps(reference_dict, indent=2))
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs_da.json", 'w') as f:
      f.write(json.dumps(ref_da_dict, indent=2))
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json", 'w') as f:
      f.write(json.dumps(prediction_dict, indent=2))
      
      
      
def predict(args):
    te_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='test')
    print('Data loaded!!!')
    
    # Load the model
    tokenizer = GPT2Tokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model.pretrain_clf = args.pretrain_clf
    model.lamb = args.lamb
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'correct': generate_label(
            batch,
            model, 
            tokenizer, 
            args,
            device='cuda:0')
        },
        batched=True, 
        batch_size=1,
    )
    
    # prepare evaluation data
    pred_list = list(test_output)
    correct, total = 0, 0
    for item in pred_list:
        correct += item['correct']
        total += 1
    print(f'Accuracy: {round(correct/total, 4)}')
    
    
    
def decode_time(args):
    te_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='test')
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
    model.pretrain_clf = args.pretrain_clf
    model.lamb = args.lamb
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
    p.add_argument('-c', '--ckpt', type=str, default="2500", 
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-05-11-07-08-10', 
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', 
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="dailydialog_raw", 
                   help="specify the dataset: dailydialog_raw, convai2_raw")
    p.add_argument('--model_name', type=str, default="microsoft/DialoGPT-medium", 
                   help="specify the model name: gpt2-medium, microsoft/DialoGPT-medium")
    p.add_argument('--pretrain_clf', action='store_true', 
                   help="specify whether pretraining a classifier or finetuning the sidenet")
    p.add_argument('--lamb', type=float, default=1.0)
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--encoder_max_length', type=int, default=200)
    p.add_argument('--decoder_max_length', type=int, default=60)
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
    elif args.task == 'pred':
        predict(args)
    elif args.task == 'dect':
        decode_time(args)
    else:
        print('Unrecognizable task!!!')