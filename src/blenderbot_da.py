# -*- coding: utf-8 -*-

import json
import argparse
import time
import os
from datetime import datetime
from utils import parse_data, batchify_data, make_batch_inputs, prepare_eval, test_ppl
from datasets import Dataset
from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import Seq2SeqLMOutput

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
    ):
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels[:,1:], self.config.pad_token_id, self.config.decoder_start_token_id
                )
            
        outputs = self.model(
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
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        
        masked_lm_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            labels = labels[:, 1:]
            masked_lm_loss = loss_fct(lm_logits.contiguous().view(-1, self.config.vocab_size), 
                                      labels.contiguous().view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
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
                    labels=inputs['labels'])
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
                          labels=inputs['labels'])
      else:
        outputs = model(input_ids=inputs['input_ids'],
                        labels=inputs['labels'])
      if has_labels:
        loss = outputs[0].mean().detach()
      else:
        loss = None
    # If we're only computing the conditional log-likelihood, return.
    if prediction_loss_only:
      return (loss, None, None)
    # Otherwise run model.generate() to get predictions.
    if isinstance(model, torch.nn.DataParallel):
      preds = model.module.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        num_beams=self.num_beams,
        max_length=self.max_length,
      )
    else:
      preds = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],
        num_beams=self.num_beams,
        max_length=self.max_length,
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
        ckpt_path = f"{args.model_name}_daily_pretrain_{args.timestamp}/checkpoint-{args.ckpt}"
        # update timestamp and create new path for ckpt
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    tokenizer = BlenderbotTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens([i for i in SPECIAL_TOKENS.values()])
    print(f"Vocab size: {len(tokenizer)}")
    
    train_data_tokenized = batchify_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_data(val_df, tokenizer, args)
    
    model = Seq2SeqModel.from_pretrained(ckpt_path)
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



def generate_sentences(batch,
                       model,
                       tokenizer,
                       args, 
                       device='cuda:0'):
  # Create batch inputs.
  features = make_batch_inputs(
      batch=batch, 
      tokenizer=tokenizer, 
      args=args, 
      device=device)
  # Generate with beam search.
  generated_ids = model.generate(
      input_ids=features['input_ids'], 
      attention_mask=features['attention_mask'],
      num_beams=args.beam_size,
      do_sample=True,
      top_k=args.top_k,
      top_p=args.top_p,
      max_length=args.max_generation_length,
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
    
    if args.timestamp == '0':
        model = Seq2SeqModel.from_pretrained(f"{args.model_name}")
    else:
        model = Seq2SeqModel.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    # Compute ppl
    ppl = test_ppl(te_df, model, tokenizer, args)
    ppl = round(ppl, 4)
    print(f"Test ppl: {ppl}")
    
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
    p.add_argument('-c', '--ckpt', type=str, default="193280", 
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-02-14-04-57-04', 
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='base', 
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="dailydialog_raw", 
                   help="specify the dataset: dailydialog_raw, convai2_raw")
    p.add_argument('--model_name', type=str, default="facebook/blenderbot-400M-distill", 
                   help="specify the model name: t5-base, facebook/blenderbot-400M-distill")
    p.add_argument('--no_da', action='store_true', 
                   help="specify whether include DA tags in inputs")
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
    elif args.task == 'dect':
        decode_time(args)
    else:
        print('Unrecognizable task!!!')