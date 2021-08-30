# -*- coding: utf-8 -*-

import torch
import json
import pandas as pd
from datasets import Dataset

# Special tokens
SPECIAL_TOKENS = {
"inform": "[inform]",
"question": "[question]",
"directive": "[directive]",
"commissive": "[commissive]",  
}

def linear_da(cxt_intents):
    out = []
    for cxt_intent in cxt_intents:
        if type(cxt_intent) is list:
            out.append(' '.join(cxt_intent))
        else:
            out.append(cxt_intent)
    return out

def parse_data(in_file='../data/dailydialog.json', mode='train'):
    with open(in_file, 'r') as f:
        data = json.load(f)[mode]
    contexted = []
    for bid, data_dict in enumerate(data):
        if 'fs' in in_file:
            row = (data_dict['dialog_id'], data_dict['utt_id'], 
                   data_dict['cxt'], data_dict['cxt_intents'],
                   data_dict['target_response'], data_dict['target_intents'])
        elif 'raw' in in_file:
            row = (data_dict['dialog_id'], data_dict['utt_id'], 
                   data_dict['cxt'], data_dict['cxt_intents'],
                   data_dict['target_response'], data_dict['target_intent'])
        else:
            row = (data_dict['dialog_id'], data_dict['utt_id'], 
                   data_dict['cxt'], linear_da(data_dict['cxt_intents']),
                   data_dict['target_response'], data_dict['target_intents'])
        contexted.append(row)
    columns = ['dialogue_id', 'utt_id', 'context_utt', 
               'context_da', 'target', 'target_da']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_profile_data(in_file='../data/convai2_raw.json', mode='train'):
    with open(in_file, 'r') as f:
        data = json.load(f)[mode]
    contexted = []
    for bid, data_dict in enumerate(data):
        row = (data_dict['dialog_id'], data_dict['utt_id'], 
               '\t'.join(data_dict['profile']), 
               '\t'.join(data_dict['cxt']), 
               data_dict['target'])
        contexted.append(row)
    columns = ['dialogue_id', 'utt_id', 'profile', 'context_utt', 'target']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_dis_kb_data(in_file='../data/dailydialog_dis.json', mode='train'):
    with open(in_file, 'r') as f:
        data = json.load(f)[mode]
    contexted = []
    for bid, data_dict in enumerate(data):
        row = (data_dict['label'], data_dict['utterance'], '\t'.join(data_dict['profile']))
        contexted.append(row)
    columns = ['label', 'utterance', 'profile']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_dis_data(in_file='../data/dailydialog_dis.json', mode='train'):
    with open(in_file, 'r') as f:
        data = json.load(f)[mode]
    contexted = []
    for bid, data_dict in enumerate(data):
        row = (data_dict['intent'], data_dict['utterance'])
        contexted.append(row)
    columns = ['intent', 'utterance']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_pred_data(file_path='../outputs/', output_file='refs.json', ref_file='da_refs.json'):
    with open(file_path + ref_file, 'r') as f:
        da_refs = json.load(f)['values']
        
    with open(file_path + output_file, 'r') as f:
        generated_list = json.load(f)['values']
        
    contexted = []
    for i in range(len(generated_list)):
        sent = generated_list[i]['generated']
        da = da_refs[i]['target_da'][0]
        row = (da, sent)
        contexted.append(row)
    columns = ['intent', 'utterance']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

def parse_pred_kb_data(file_path='../outputs/', output_file='refs.json', ref_file='da_refs.json'):
    with open(file_path + ref_file, 'r') as f:
        kb_refs = json.load(f)['values']
        
    with open(file_path + output_file, 'r') as f:
        generated_list = json.load(f)['values']
        
    contexted = []
    for i in range(len(generated_list)):
        sent = generated_list[i]['generated']
        kb = '\t'.join(kb_refs[i]['profile'])
        row = (1, sent, kb)
        contexted.append(row)
    columns = ['label', 'utterance', 'profile']
    data_df = pd.DataFrame.from_records(contexted, columns=columns)
    return data_df

# Specific to dataset.
def construct_input_for_batch(tokenizer, batch, args):
    if args.dataset in ["dailydialog", "dailydialog_raw", "dailydialog_100_raw", "dailydialog_500_raw", 
                          "dailydialog_1000_raw", "dailydialog_5000_raw", "dailydialog_10000_raw"]:
        source, target = [], []
        if 'gpt' in args.model_name.lower() and args.task in ['train', 'ft', 'pred']:
            for i in range(len(batch['target'])):
                inp = ""
                for cur_utt in batch['context_utt'][i]:
                    inp += f"{cur_utt} {tokenizer.eos_token} "
                utt = batch['target'][i]
                inp += f"{utt} {tokenizer.eos_token} "
                da = batch['target_da'][i]
                da_label = f"{SPECIAL_TOKENS[da]} "
                source.append(inp.strip())
                target.append(da_label.strip())
        else:
            for utt, da in zip(batch['context_utt'], batch['context_da']):
                inp = ""
                for cur_utt, cur_da in zip(utt, da):
                    inp += f"{cur_utt} {tokenizer.eos_token} "
                source.append(inp.strip())
            
            for da, utt in zip(batch['target_da'], batch['target']):
                out = ""
                if not args.no_da:
                    out += f"{SPECIAL_TOKENS[da]} "
                out += f"{utt} {tokenizer.eos_token} "
                target.append(out.strip())
    
        if batch['dialogue_id'][0] == 0:
            print(source[0])
            print(target[0])
            print()
        return source, target
    elif args.dataset in ['convai2_raw', 'convai2_100_raw', 'convai2_500_raw', 
                          'convai2_1000_raw', 'convai2_5000_raw', 'convai2_10000_raw']:
        source, target, kb = [], [], []
        if 'gpt' in args.model_name.lower() and args.task in ['train', 'ft']:
            for i in range(len(batch['target'])):
                inp = ""
                context_utts = batch['context_utt'][i].split('\t')
                for cur_utt in context_utts:
                    inp += f"{cur_utt} {tokenizer.eos_token} "
                utt = batch['target'][i]
                inp += f"{utt} {tokenizer.eos_token} "
                profiles = batch['profile'][i].split('\t')
                profile = ""
                for item in profiles:
                    profile += f"{item} {tokenizer.eos_token} "
                source.append(inp.strip())
                target.append(inp.strip())
                kb.append(profile.strip())
        else:
            for i in range(len(batch['target'])):
                inp = ""
                context_utts = batch['context_utt'][i].split('\t')
                for cur_utt in context_utts:
                    inp += f"{cur_utt} {tokenizer.eos_token} "
                utt = batch['target'][i]
                out = f"{utt} {tokenizer.eos_token} "
                profiles = batch['profile'][i].split('\t')
                profile = ""
                for item in profiles:
                    profile += f"{item} {tokenizer.eos_token} "
                source.append(inp.strip())
                target.append(out.strip())
                kb.append(profile.strip())
    
        if batch['dialogue_id'][0] == 0:
            print(source[0])
            print(target[0])
            print(kb[0])
            print()
        return source, target, kb
    elif args.dataset in ["dailydialog_dis",]:
        source, target = [], []
        for utt, da in zip(batch['utterance'], batch['intent']):
            inp = utt.strip()
            source.append(inp)
            target.append(da.strip())
        return source, target
    elif args.dataset in ["convai2_dis",]:
        source, target, kb = [], [], []
        for i in range(len(batch['utterance'])):
            label = batch['label'][i]
            utt = batch['utterance'][i]
            profiles = batch['profile'][i].split('\t')
            profile = ""
            for item in profiles:
                profile = f"{item} {tokenizer.eos_token} "
            source.append(utt.strip())
            kb.append(profile.strip())
            target.append(label)
        return source, target, kb
    else:
        raise NotImplementedError("Please add a processor for this dataset.")

def make_batch_inputs(batch, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  input_lists, _ = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  return batch_features

def make_batch_data(batch, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  input_lists, label_list = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_labels = tokenizer(label_list, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
  return batch_features, batch_labels

def make_batch_kb_data(batch, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  input_list, _, kb_list = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_list, padding=True, return_tensors='pt')
  batch_profiles = tokenizer(kb_list, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_profiles = dict([(k, v.to(device)) for k, v in batch_profiles.items()])
  return batch_features, batch_profiles

def batch_tokenize(dataset_batch, tokenizer, args):
  source, target = construct_input_for_batch(tokenizer, dataset_batch, args)
  res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
         "attention_mask": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["attention_mask"],
          "labels": tokenizer(
              target,
              padding='max_length', 
              truncation=True,
              max_length=args.decoder_max_length
          )["input_ids"],
  }
  return res

def batchify_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def batch_kb_tokenize(dataset_batch, tokenizer, args):
  source, target, kb = construct_input_for_batch(tokenizer, dataset_batch, args)
  res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
          "decoder_profile": tokenizer(
              kb,
              padding='max_length', 
              truncation=True,
              max_length=args.decoder_max_length
          )["input_ids"], 
          "labels": tokenizer(
              target,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
  }
  return res

def batchify_kb_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_kb_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def batch_dis_tokenize(dataset_batch, tokenizer, args):
  source, target = construct_input_for_batch(tokenizer, dataset_batch, args)
  labels = ["inform", "question", "directive", "commissive"]
  res = {
          "input_ids": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.encoder_max_length
          )["input_ids"],
          "labels": [labels.index(t) for t in target],
  }
  return res

def batchify_dis_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_dis_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def batch_dis_kb_tokenize(dataset_batch, tokenizer, args):
  source, target, kb = construct_input_for_batch(tokenizer, dataset_batch, args)
  res = {
          "input_ids": tokenizer(
              kb,
              padding='max_length', 
              truncation=True,
              max_length=args.decoder_max_length
          )["input_ids"],
          "generated_utt": tokenizer(
              source,
              padding='max_length', 
              truncation=True,
              max_length=args.max_generation_length
          )["input_ids"], 
          "labels": target,
  }
  return res

def batchify_dis_kb_data(df, tokenizer, args):
  dataset = Dataset.from_pandas(df)
  data_tokenized = dataset.map(
    lambda batch: batch_dis_kb_tokenize(batch, tokenizer, args),
    batched=True
  )
  return data_tokenized

def compute_loss(batch, model, tokenizer, args):
  batch_feature, batch_label = make_batch_data(batch, tokenizer, args)
  with torch.no_grad():
    outputs = model(input_ids=batch_feature['input_ids'],
                    labels=batch_label['input_ids'],
                    decoder_da_inputs=batch_label['input_ids'][:,0],
                    decoder_dis_loss=False)
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def compute_loss_gpt2(batch, model, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  args.task = 'train'
  input_lists, label_lists = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_labels = tokenizer(label_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
  with torch.no_grad():
    # Note that the labels are shifted inside the model, so set labels = input_ids
    outputs = model(input_ids=batch_features['input_ids'],
                    labels=batch_features['input_ids'],
                    attention_mask=batch_features['attention_mask'],
                    decoder_da_inputs=batch_labels['input_ids'][:,0],
                    decoder_dis_loss=False)
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl_gpt2(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss_gpt2(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def compute_loss_gpt2_kb(batch, model, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  args.task = 'train'
  input_lists, _, kb_lists = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_profiles = tokenizer(kb_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_profiles = dict([(k, v.to(device)) for k, v in batch_profiles.items()])
  with torch.no_grad():
    # Note that the labels are shifted inside the model, so set labels = input_ids
    outputs = model(input_ids=batch_features['input_ids'],
                    labels=batch_features['input_ids'],
                    decoder_profile=batch_profiles['input_ids'])
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl_gpt2_kb(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss_gpt2_kb(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def compute_loss_kb(batch, model, tokenizer, args, device='cuda:0'):
  # Concatenate the concept names for each example in the batch.
  args.task = 'train'
  input_lists, label_lists, kb_lists = construct_input_for_batch(tokenizer, batch, args)
  # Use the model's tokenizer to create the batch input_ids.
  batch_features = tokenizer(input_lists, padding=True, return_tensors='pt')
  batch_labels = tokenizer(label_lists, padding=True, return_tensors='pt')
  batch_profiles = tokenizer(kb_lists, padding=True, return_tensors='pt')
  # Move all inputs to the device.
  batch_features = dict([(k, v.to(device)) for k, v in batch_features.items()])
  batch_labels = dict([(k, v.to(device)) for k, v in batch_labels.items()])
  batch_profiles = dict([(k, v.to(device)) for k, v in batch_profiles.items()])
  with torch.no_grad():
    # Note that the labels are shifted inside the model, so set labels = input_ids
    outputs = model(input_ids=batch_features['input_ids'],
                    decoder_input_ids=batch_labels['input_ids'][:,:-1],
                    labels=batch_labels['input_ids'],
                    decoder_profile=batch_profiles['input_ids'])
    eval_loss = outputs.loss.item()
  return [eval_loss] 

def test_ppl_kb(val_df, model, tokenizer, args):
  loss_dict = Dataset.from_pandas(val_df).map(
    lambda batch: {'loss': compute_loss_kb(batch, model, tokenizer, args)},
    batched=True,
    batch_size=1,
  )
  
  eval_loss = 0.
  nb_eval_steps = 0
  for item in list(loss_dict):
      eval_loss += item['loss']
      nb_eval_steps += 1
  eval_loss = eval_loss / nb_eval_steps
  ppl = torch.exp(torch.tensor(eval_loss))
  return ppl.item()

def prepare_eval(output_list):
    ref_list, da_list, pred_list = [], [], []
    for item in output_list:
        pred_list.append({"generated": item['generated']})
        ref_list.append({"target": [item['target']]})
        da_list.append({"target_da": [item['target_da']]})
    return ref_list, pred_list, da_list

def prepare_eval_kb(output_list):
    ref_list, da_list, pred_list = [], [], []
    for item in output_list:
        pred_list.append({"generated": item['generated']})
        ref_list.append({"target": [item['target']]})
        da_list.append({"profile": item['profile']})
    return ref_list, pred_list, da_list
