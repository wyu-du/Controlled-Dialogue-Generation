# Controlled-Dialogue-Generation
This repository contains the data and code for the paper ["SideControl: Controlled Open-domain Dialogue Generation via Additive Side Networks"](https://aclanthology.org/2021.findings-emnlp.188/) (EMNLP2021-Findings).


## Environment
Under the server environment of `python=3.6` and `CUDA 11.1`, install the following packages:
```
pip install -r requirements.txt
```

## Training
Training the DialoGPT-SideNet on `DailyDialog` full training set includes two steps. 
First, pretrain a DA classifier:
```
python gpt2_da_sidenet.py -d dailydialog_raw -t train -f base --pretrain_clf
```
Second, train a SideNet (remember to replace the `timestamp` and `ckpt` with your own model checkpoint):
```
python gpt2_da_sidenet.py -d dailydialog_raw -t ft -f sidenet --timestamp 2021-05-11-07-08-10 --ckpt 2500
```


Training the DialoGPT-SideNet on `ConvAI2` full training set includes one step:
```
python gpt2_kb_sidenet.py -d convai2_raw -t train -f sidenet
``` 

## Decoding
Get the DialoGPT-SideNet predictions on `DailyDialog` full testing set (remember to replace the `timestamp` and `ckpt` with your own model checkpoint):
```
python gpt2_da_sidenet.py -d dailydialog_raw -t eval -f sidenet --timestamp 2021-05-11-07-08-10 --ckpt 2500
```

Get the DialoGPT-SideNet predictions on `ConvAI2` full testing set (remember to replace the `timestamp` and `ckpt` with your own model checkpoint):
```
python gpt2_kb_sidenet.py -d convai2_raw -t eval -f sidenet --timestamp 2021-04-26-10-21-06 --ckpt 47839
```


## Evaluation
Compute text quality metrics for DialoGPT-SideNet predictions:
```
python evaluation.py \
--mode sent \
--reference_file {args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs.json \
--output_file {args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json
```

Compute text controllability metrics (knowledge document control) for DialoGPT-SideNet predictions:
```
python evaluation.py \
--mode kb \
--reference_file {args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/refs.json \
--output_file {args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json
```

Compute text controllability metrics (semantic label control) for DialoGPT-SideNet predictions.
First, train an independent DA classifier:
```
python bert_da_eval.py -d dailydialog_dis -t train -f clf
```
Second, compute the accuracy predicted by the independent DA classifier (remember to replace the `timestamp`, `ckpt` and `output_file` accordingly):
```
python bert_da_eval.py -d dailydialog_dis -t pred -f clf \
--timestamp 2021-04-11-05-57-18 --ckpt 10000 \
--output_file {args.model_name}_{args.dataset}_{args.flag}_{args.timestamp}/outs.json
```

## Citation
Please cite our work if you are interested.
```
@inproceedings{du-ji-2021-sidecontrol-controlled,
    title = "{S}ide{C}ontrol: Controlled Open-domain Dialogue Generation via Additive Side Networks",
    author = "Du, Wanyu  and
      Ji, Yangfeng",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2021",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-emnlp.188",
    pages = "2175--2194",
}
```