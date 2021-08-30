from nltk.util import ngrams
from nltk import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
import numpy as np
import json
import argparse
import torch
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import gensim.downloader as api

def compute_bleu(references, candidates):
    ref_list, dec_list = [], []
    for i in range(len(candidates)):
        dec_list.append(word_tokenize(candidates[i]))
        if type(references[i]) is list:
            tmp = []
            for ref in references[i]:
                tmp.append(word_tokenize(ref))
            ref_list.append(tmp)
        else:
            ref_list.append([word_tokenize(references[i])])
    bleu1 = corpus_bleu(ref_list, dec_list,
                        weights=(1, 0, 0, 0))
    bleu2 = corpus_bleu(ref_list, dec_list,
                        weights=(0, 1, 0, 0))
    print("BLEU-1: "+str(round(bleu1,4)))
    print("BLEU-2: "+str(round(bleu2,4)))
    
def compute_ppl(candidates):
    model = OpenAIGPTLMHeadModel.from_pretrained("openai-gpt").to('cuda:0')
    tokenizer = OpenAIGPTTokenizer.from_pretrained("openai-gpt")
    
    lls = []
    for i in range(len(candidates)):
        inputs = tokenizer(candidates[i], return_tensors='pt')
        inputs = dict([(k, v.to('cuda:0')) for k, v in inputs.items()])
        with torch.no_grad():
            if inputs["input_ids"].size(-1) > 1:
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
        lls.append(loss)
    ppl = np.exp(np.mean(lls))
    print("GPT Perplexity: "+str(round(ppl,4)))
    
def compute_meteor(references, candidates):
    score_list = []
    for i in range(len(candidates)):
        if type(references[i]) is list:
            ref_list = references[i]
        else:
            ref_list = [references[i]]
        score = meteor_score(ref_list, candidates[i])
        score_list.append(score)
    print("METEOR: "+str(round(np.mean(score_list), 4)))
    
def distinct_ngram(candidates, n=2):
    """Return basic ngram statistics, as well as a dict of all ngrams and their freqsuencies."""
    ngram_freqs = {}   # ngrams with frequencies
    ngram_len = 0  # total number of ngrams
    for candidate in candidates: 
        for ngram in ngrams(word_tokenize(candidate), n):
            ngram_freqs[ngram] = ngram_freqs.get(ngram, 0) + 1
            ngram_len += 1
    # number of unique ngrams
    uniq_ngrams = len([val for val in ngram_freqs.values() if val == 1])
    distinct_ngram = len(ngram_freqs) / ngram_len if ngram_len > 0 else 0
    print(f'Distinct {n}-grams:', round(distinct_ngram,4))
    return ngram_freqs, uniq_ngrams, ngram_len

def knowledge_f1(references, candidates):
    """
    This function is copied from: https://github.com/PaddlePaddle/Research/blob/master/NLP/Dialogue-PLATO/tools/knowledge_f1.py
    """
    cnt = 0
    res = 0.0
    r = 0.0
    p = 0.0
    stopwords = set()
    with open("../data/stopwords.txt") as f:
        for line in f:
            word = line.strip()
            stopwords.add(word)
    
    for candidate, reference in zip(candidates, references):
        cnt += 1
        if type(reference) is list:
            reference = reference[0]
        knowledges = reference.strip().split('\t')

        words = set()
        for sent in knowledges:
            for word in sent.split():
                words.add(word.lower())
        words = words - stopwords
        k_len = len(words)
        
        pred = set()
        for word in candidate.split():
            pred.add(word.lower())
        pred = pred - stopwords
        pred_len = len(pred)
        overlap = len(words & pred)

        if overlap == 0:
            continue

        recall = float(overlap) / k_len
        r += recall
        precison = float(overlap) / pred_len
        p += precison
        res += 2*recall*precison/(recall+precison)
    print(f"Recall:{r/cnt}")
    print(f"Precison:{p/cnt}")
    print(f"F1:{res/cnt}")
    print("Recall/Precision/F1:{:0,.4f}/{:0,.4f}/{:0,.4f}".format(r/cnt, p/cnt, res/cnt))
    
def compute_cos_sim(references, candidates):
    # load pre-trained word-vectors from gensim-data
    word_vectors = api.load("glove-wiki-gigaword-100")
    vocab_list = list(word_vectors.vocab.keys())
    
    stopwords = set()
    with open("../data/stopwords.txt") as f:
        for line in f:
            word = line.strip()
            stopwords.add(word)
    
    sim_list = []
    for i in range(len(candidates)):
        dec_set = set()
        for word in word_tokenize(candidates[i]):
            word = word.lower()
            if word in vocab_list:
                dec_set.add(word)
        dec_set = dec_set - stopwords
        dec_list = list(dec_set)
        if len(dec_list) == 0: continue
    
        ref_set = set()
        for word in word_tokenize(references[i]):
            word = word.lower()
            if word in vocab_list:
                ref_set.add(word)
        ref_set = ref_set - stopwords
        ref_list = list(ref_set)
        # compute cosine similarity between two sets of docvecs from the trained set
        cos_sim = word_vectors.n_similarity(dec_list, ref_list)
        sim_list.append(cos_sim)
        
    avg_sim = np.mean(sim_list)
    print("COS-Simlarity (set): "+str(round(avg_sim,4)))
    
def read_data(ref_file, in_file, mode='kb'):
    references, candidates = [], []
    if mode == 'kb':
        with open('../outputs/'+ref_file, 'r') as f:
            ref_list = json.load(f)['values']
            for ref in ref_list:
                references.append(ref['profile'])
    else:
        with open('../outputs/'+ref_file, 'r') as f:
            ref_list = json.load(f)['values']
            for ref in ref_list:
                references.append(ref['target'][0])
    with open('../outputs/'+in_file, 'r') as f:
        out_list = json.load(f)['values']
        for out in out_list:
            candidates.append(out['generated'])
    return references, candidates

    
if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-m', '--mode', type=str, default="sent")
    p.add_argument('-r', '--reference_file', type=str, default="refs.json")
    p.add_argument('-o', '--output_file', type=str, default="blenderbot-400M-distill_dailydialog_raw_zero_0_outs.json")
    args = p.parse_args()
    
    # Read Data
    references, candidates = read_data(ref_file = args.reference_file,
                                       in_file = args.output_file,
                                       mode = args.mode)
#    v_len = int(0.8*len(candidates))
#    references = references[v_len:]
#    candidates = candidates[v_len:]
    
    # Compute metrics
    if args.mode == 'kb':
        knowledge_f1(references, candidates)
        compute_cos_sim(references, candidates)
    else:
        compute_bleu(references, candidates)
        compute_meteor(references, candidates)
        distinct_ngram(candidates, n=1)
        distinct_ngram(candidates, n=2)
#        compute_ppl(candidates)
    