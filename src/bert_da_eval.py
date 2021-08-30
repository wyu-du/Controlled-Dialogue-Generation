# -*- coding: utf-8 -*-

import json
import argparse
import time
import os
from utils import parse_dis_data, batchify_dis_data, make_batch_inputs, parse_pred_data
from datasets import Dataset
from transformers import BertTokenizer, PreTrainedModel, BertModel, BertConfig
from transformers import Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert
import torch
from torch.nn import functional as F
from torch import nn



class BertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



class BertForSequenceClassification(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 4

        self.bert = BertModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.init_weights()
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. 
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`. 
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Extract features from base model
        with torch.no_grad():
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        sequence_output = outputs[0]
        
        ones = torch.ones_like(input_ids).type_as(sequence_output)
        zeros = torch.zeros_like(input_ids).type_as(sequence_output)
        mask = torch.where(input_ids==self.config.pad_token_id, zeros, ones)
        mask = mask.type_as(sequence_output)
        sequence_output = sequence_output * mask.unsqueeze(2)
        sequence_output = sequence_output.sum(dim=1)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        
        # Freeze the base model
        for param in self.bert.parameters():
            param.requires_grad = False

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
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
    # We don't use .loss here since the model may return tuples instead of ModelOutput.
    return outputs["loss"] if isinstance(outputs, dict) else outputs[0]



def train(args):
    # Load the dataset
    trn_df = parse_dis_data(in_file=f'../data/{args.dataset}.json', mode='train')
    val_df = parse_dis_data(in_file=f'../data/{args.dataset}.json', mode='validation')
    
    # Load the pre-trained model
    ckpt_path = None
    if args.task == 'train':
        ckpt_path = args.model_name
    else:
        ckpt_path = f"{args.model_name}_daily_pretrain_{args.timestamp}/checkpoint-{args.ckpt}"
        # update timestamp and create new path for ckpt
        args.timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
    
    tokenizer = BertTokenizer.from_pretrained(ckpt_path)
    print(f"Vocab size: {len(tokenizer)}")
    
    train_data_tokenized = batchify_dis_data(trn_df, tokenizer, args)
    valid_data_tokenized = batchify_dis_data(val_df, tokenizer, args)
    
    model = BertForSequenceClassification.from_pretrained(ckpt_path)
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
        warmup_steps=100,
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
  labels = ["inform", "question", "directive", "commissive"]
  # Create batch inputs.
  features = make_batch_inputs(
      batch=batch, 
      tokenizer=tokenizer, 
      args=args, 
      device=device)
  # Generate preds.
  outputs = model(input_ids=features['input_ids'])
  logits = F.log_softmax(outputs.logits, dim=-1)
#  print(logits)
  preds = torch.argmax(logits, dim=-1)
#  print(preds)
  outs = [labels[pred.item()] for pred in preds]
  print(outs)
  return outs



def test(args):
    te_df = parse_dis_data(in_file=f'../data/{args.dataset}.json', mode='test')
    print('Data loaded!!!')
    
    # Load the model
    if args.timestamp == '0':
        tokenizer = BertTokenizer.from_pretrained(f"{args.model_name}")
    else:
        tokenizer = BertTokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    if args.timestamp == '0':
        model = BertForSequenceClassification.from_pretrained(f"{args.model_name}")
    else:
        model = BertForSequenceClassification.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'generated': generate_label(
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
    pred_list = list(test_output)
    prediction_dict = {
        "language": "en",
        "values": pred_list,  
    }
    
    correct, total = 0, 0
    for item in pred_list:
        if item['intent'] == item['generated']:
            correct += 1
        total += 1
    print(f'Accuracy: {round(correct/total, 4)}')
    
    if args.timestamp == '0':
        os.makedirs(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}")
    
    with open(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json", 'w') as f:
      f.write(json.dumps(prediction_dict, indent=2))



def predict(args):
    te_df = parse_pred_data(file_path='../outputs/', 
                            output_file=args.output_file,
                            ref_file=args.ref_file)
    print('Data loaded!!!')
    
    # Load the model
    tokenizer = BertTokenizer.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = BertForSequenceClassification.from_pretrained(f"{args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/checkpoint-{args.ckpt}")
    model = model.to('cuda:0')
    print('Model loaded!!!')
    
    # Make predictions
    test_output = Dataset.from_pandas(te_df).map(
        lambda batch: {'predicted': generate_label(
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
    pred_list = list(test_output)
    prediction_dict = {
        "language": "en",
        "values": pred_list,  
    }
    
    correct, total = 0, 0
    for item in pred_list:
        if item['intent'] == item['predicted']:
            correct += 1
        total += 1
    print(f'Accuracy: {round(correct/total, 4)}')
    
    if not os.path.exists('../outputs/da_preds/'):
        os.makedirs('../outputs/da_preds/')
    out_file = '../outputs/da_preds/' + args.output_file
    with open(out_file, 'w') as f:
      f.write(json.dumps(prediction_dict, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-t', '--task', type=str, default="pred", 
                   help="specify the task to do: (train)ing, ft(finetune), (eval)uation, (pred)iction")
    p.add_argument('-c', '--ckpt', type=str, default="10000", 
                   help="Model checkpoint")
    p.add_argument('-time', '--timestamp', type=str, default='2021-04-11-05-57-18', 
                   help="Model checkpoint")
    p.add_argument('-f', '--flag', type=str, default='sent', 
                   help="Model checkpoint")
    p.add_argument('-d', '--dataset', type=str, default="dailydialog_dis")
    p.add_argument('-o', '--output_file', type=str, default="refs.json")
    p.add_argument('-r', '--ref_file', type=str, default="da_refs.json")
    p.add_argument('--model_name', type=str, default="bert-base-uncased")
    p.add_argument('-bz', '--batch_size', type=int, default=64)
    p.add_argument('--encoder_max_length', type=int, default=128)
    p.add_argument('--decoder_max_length', type=int, default=60)
    p.add_argument('--max_generation_length', type=int, default=60)
    p.add_argument('--beam_size', type=int, default=4)
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
    else:
        predict(args)