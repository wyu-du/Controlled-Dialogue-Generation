# -*- coding: utf-8 -*-

import json
import argparse
import time
from datetime import datetime
from utils import parse_data, batchify_data, make_batch_data, test_ppl, prepare_eval
from datasets import Dataset
from transformers import BlenderbotTokenizer, BlenderbotModel, BlenderbotForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import Seq2SeqLMOutput, SequenceClassifierOutput

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



# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids



class Seq2SeqModel(BlenderbotForConditionalGeneration):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.model_dim = config.d_model
        self.base = BlenderbotModel.from_pretrained("facebook/blenderbot-400M-distill")
        self.side_init()
        self.num_labels = 4
        self.classifier = nn.Linear(self.model_dim, self.num_labels)
        self.alpha = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.lm_head = nn.Linear(config.d_model, self.base.shared.num_embeddings, bias=False)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.base.shared.num_embeddings)))
        
    def side_init(self):
        self.side_emb = nn.Embedding(4, self.model_dim)
        self.intent_layer = nn.Linear(self.model_dim, self.model_dim)
        self.W1 = nn.Linear(self.model_dim*2, self.model_dim, bias=False)
        
    def get_pad_mask(self, labels):
        ones = torch.ones_like(labels)
        zeros = torch.zeros_like(labels)
        mask = torch.where(labels==self.config.pad_token_id, zeros, ones)
        return mask
        
    def side_net(self, intents, base_output):
        intents = intents.unsqueeze(1) # B x 1
        intent_emb = self.side_emb(intents) # B x 1 x E
        intent_emb = self.intent_layer(intent_emb) # B x 1 x D
        intent_emb = intent_emb.repeat(1, base_output.size(1), 1) #  B x S x D
        inp_emb = torch.cat([intent_emb, base_output], dim=2) # B x S x 2D
        h_1 = self.W1(inp_emb) # B x S x D
        h_3 = torch.tanh(h_1) # B x S x D
        return h_3
        
    def get_encoder(self):
        return self.base.get_encoder()

    def get_decoder(self):
        return self.base.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
        
    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        decoder_input_ids = torch.ones((input_ids.size(0), 1), 
                                       device=torch.device('cuda:0'), 
                                       dtype=torch.long)
        decoder_input_ids = decoder_input_ids * self.config.decoder_start_token_id
        if past is not None:
            decoder_input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "decoder_input_ids": decoder_input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "decoder_da_inputs": kwargs['decoder_da_inputs']
        }

    def adjust_logits_during_generation(self, logits, cur_len, max_length):
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_id_to_be_generated(logits, self.config.eos_token_id)
        return logits

    @staticmethod
    def _force_token_id_to_be_generated(scores, token_id) -> None:
        """force one of token_ids to be generated by setting prob of all other tokens to 0 (logprob=-float("inf"))"""
        scores[:, [x for x in range(scores.shape[1]) if x != token_id]] = -float("inf")

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        decoder_args=None,
        decoder_da_inputs=None,
        decoder_dis_loss=True,
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        intents = decoder_da_inputs
        if intents[0] > self.base.shared.num_embeddings:
            intents -= self.base.shared.num_embeddings+1

        pad_mask = None
        if labels is not None:
            labels = labels[:, 1:]
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
            labels.masked_fill_(labels == -100, self.config.pad_token_id)
            pad_mask = self.get_pad_mask(labels)
        
        # Extract features from base model
        with torch.no_grad():
            base_outputs = self.base(
                input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                encoder_outputs=encoder_outputs,
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                decoder_inputs_embeds=decoder_inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        
        base_output = base_outputs.last_hidden_state
        
        if not self.pretrain_clf:
            # get side outputs
            side_output = self.side_net(intents, base_output) # B x S x D
            # fuse base and side output
            sequence_output = base_output * torch.sigmoid(self.alpha) + side_output * (1 - torch.sigmoid(self.alpha))
            lm_logits = self.lm_head(sequence_output) + self.final_logits_bias
        
        
        # compute classification logits
        if decoder_dis_loss and pad_mask is not None:
            if self.pretrain_clf:
                sequence_output = base_output
            else:
                sequence_output = side_output
            pad_mask = pad_mask.type_as(sequence_output)
            seq_len = pad_mask.sum(dim=1, keepdim=True)          # B x 1
            clf_output = sequence_output * pad_mask.unsqueeze(2) # B x S x D
            clf_output = clf_output.sum(dim=1)                   # B x D
            clf_output = clf_output / (seq_len + 1e-10)          # B x D
            clf_logits = self.classifier(clf_output)   
        
        # Freeze the base model
        for param in self.base.parameters():
            param.requires_grad = False
        # Freeze the classifier
        if not self.pretrain_clf:
            for param in self.classifier.parameters():
                param.requires_grad = False
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(reduce=False)
            if not self.pretrain_clf:
                loss = loss_fct(lm_logits.contiguous().view(-1, self.config.vocab_size), 
                                          labels.contiguous().view(-1))
                loss = loss * pad_mask.contiguous().view(-1)
                loss = loss.mean()
            
            if decoder_dis_loss:
                dis_loss = loss_fct(clf_logits.view(-1, self.num_labels), intents.view(-1))
                dis_loss = dis_loss.mean()
                if self.pretrain_clf:
                    loss = dis_loss
                else:
                    loss = loss + self.lamb * dis_loss
                    
        if self.pretrain_clf:
            if not return_dict:
                output = (clf_logits,) + base_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                return SequenceClassifierOutput(
                    loss=loss,
                    logits=clf_logits,
                )
        else:
            if not return_dict:
                output = (lm_logits,) + base_outputs[1:]
                return ((loss,) + output) if loss is not None else output
            else:
                return Seq2SeqLMOutput(
                    loss=loss,
                    logits=lm_logits,
                    past_key_values=base_outputs.past_key_values,
                    decoder_hidden_states=base_outputs.decoder_hidden_states,
                    decoder_attentions=base_outputs.decoder_attentions,
                    cross_attentions=base_outputs.cross_attentions,
                    encoder_last_hidden_state=base_outputs.encoder_last_hidden_state,
                    encoder_hidden_states=base_outputs.encoder_hidden_states,
                    encoder_attentions=base_outputs.encoder_attentions,
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
                    labels=inputs['labels'],
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
                          labels=inputs['labels'],
                          decoder_da_inputs=inputs['labels'][:,0])
      else:
        outputs = model(input_ids=inputs['input_ids'],
                        labels=inputs['labels'],
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
        num_beams=self.num_beams,
        max_length=self.max_length,
        **model_kwargs
      )
    else:
      preds = model.generate(
        input_ids=inputs['input_ids'], 
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
      labels = inputs.get('labels')
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
    
    tokenizer = BlenderbotTokenizer.from_pretrained(args.model_name)
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
  Function to generate outputs from a model with beam search decoding. 
  """
  # Create batch inputs.
  features, labels = make_batch_data(batch, tokenizer, args, device)
  da_ids = labels['input_ids'][:, 0]
  # Generate preds.
  with torch.no_grad():
      outputs = model(input_ids=features['input_ids'],
                      labels=labels['input_ids'][:, 1:],
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
  generated_ids = model.generate(
      input_ids=features['input_ids'], 
      num_beams=args.beam_size,
      do_sample=True,
      top_k=args.top_k,
      top_p=args.top_p,
      max_length=args.max_generation_length,
      **model_kwargs
  )
  # Use model tokenizer to decode to text.
  generated_sentences = [
      tokenizer.decode(gen_ids.tolist(), skip_special_tokens=True)
      for gen_ids in generated_ids
  ]
  print(generated_sentences[0])
  return generated_sentences



def test(args):
    te_df = parse_data(in_file=f'../data/{args.dataset}.json', mode='test')
    print('Data loaded!!!')
    
    # Load the model
    tokenizer = BlenderbotTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens([i for i in SPECIAL_TOKENS.values()])
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
    ppl = test_ppl(te_df, model, tokenizer, args)
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
    tokenizer = BlenderbotTokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model.method = args.method
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
        tokenizer = BlenderbotTokenizer.from_pretrained(f"{args.model_name}")
    else:
        tokenizer = BlenderbotTokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
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
    p.add_argument('-c', '--ckpt', type=str, default="17892", 
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-04-15-11-07-11', 
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', 
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="dailydialog_raw", 
                   help="specify the dataset: dailydialog_raw, convai2_raw")
    p.add_argument('--model_name', type=str, default="facebook/blenderbot-400M-distill", 
                   help="specify the model name: t5-base, facebook/blenderbot-400M-distill")
    p.add_argument('--pretrain_clf', action='store_true', 
                   help="specify whether pretraining a classifier or finetuning the sidenet")
    p.add_argument('--lamb', type=float, default=1.0)
    p.add_argument('-bz', '--batch_size', type=int, default=4)
    p.add_argument('--encoder_max_length', type=int, default=128)
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