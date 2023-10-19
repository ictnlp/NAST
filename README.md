# Non-autoregressive Streaming Transformer for Simultaneous Translation (NAST)

Implementation for the EMNLP 2023 paper "**[``Non-autoregressive Streaming Transformer for Simultaneous Translation``](https://openreview.net/forum?id=gzRBs4gIbz)**".

**Abstract**: We introduce a non-autoregressive streaming model for simultaneous translation.

![image](https://github.com/ictnlp/NAST/blob/main/nast.pdf)

**Highlights**: 
* NAST **outperforms SOTA autoregressive SiMT model across all latency settings** on WMT15 DE-EN dataset.
* NAST **demonstrates exceptional performance under extremely low latency conditions**. (29.82 BLEU @1.89 AL)

**Files**:

- We mainly provide the following files as plugins into  [``fairseq:5175fd``](https://github.com/pytorch/fairseq/tree/5175fd5c267adceec9445bf067597686e159e7e7) in the [``NAST``](https://github.com/ictnlp/NAST/tree/main/NAST) directory.

   ```
   NAST
   └── criterions
   │   ├── __init__.py
   │   ├── nat_loss_ngram_glat_simul.py              
   │   └── utilities.py                          
   └── generators
   │   ├── prefix_beam_search.py
   │   ├── prefix_beam_search_logits.py
   │   ├── streaming_generator.py       
   │   └── streaming_generator_chunk_wait_k.py
   └── models
   │   ├── torch_imputer
   │   │    ├── __init__.py
   │   │    ├── best_alignment.cu
   │   │    ├── imputer.cpp
   │   │    ├── imputer.cu  
   │   │    └── imputer.py
   │   ├── __init__.py 
   │   └── nonautoregressive_streaming_transformer.py
   └── scripts 
   │   ├── average_checkpoints.py
   │   └── generate_streaming.py
   └── tasks 
   │   ├── __init__.py 
   │   └── translation_ctc_streaming.py
   └── __init__.py
   ```
- We also provide all the training & test scripts in the [``shell_scripts``](https://github.com/ictnlp/NAST/tree/main/shell_scripts) directory.



**Below is a guide to replicate the results reported in the paper. We give an example of experiments on WMT15 De-En dataset.**
## Requirements & Installation
### Requirements
* Python >= 3.7
* Pytorch == 1.10.1 (tested with cuda == 11.3)

### Installation
* ``git clone --recurse-submodules https://github.com/ictnlp/NAST.git``
* ``cd NAST && cd fairseq``
* ``pip install --editable ./``
* ``python setup.py build develop``


## Stage-1 training

At the Stage-1 training, we use a batch size of approximating 64k tokens **(GPU number * max_tokens * update_freq == 64k)**.


* Set ``wait_until`` to control ``Chunk Wait-k Strategy``.
* Run the following script for Stage-1 training. (The scripts can be also found in the [``shell_scripts``](https://github.com/ictnlp/NAST/tree/main/shell_scripts) directory.)

```bash
exp=your_exp_name
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/save_checkpoint
plugin_path=/path/to/NAST_plugins
wait_until=0

fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --task translation_ctc_streaming \
    --criterion nat_loss_ngram_glat_simul --left-pad-source --glat-p 0.5:0.3@200k \
    --src-embedding-copy \
    --src-upsample-ratio 3 --plain-ctc --wait-until ${wait_until} --latency-factor 0 \
    --arch nonautoregressive_streaming_transformer \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0005 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 10000 \
    --warmup-init-lr '1e-07' --label-smoothing 0.01 \
    --dropout 0.3 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --apply-bert-init \
    --log-format 'simple' --log-interval 100 \
    --fixed-validation-seed 7 \
    --max-tokens 4096 \
    --update-freq 4 \
    --save-interval-updates 10000 \
    --keep-interval-updates 5 --keep-last-epochs 5 \
    --max-update 300000
```

## Stage-2 Training

At the Stage-2 training, we use a batch size of approximating 256k tokens **(GPU number * max_tokens * update_freq == 256k)**.

* Set ``wait_until`` to control ``Chunk Wait-k Strategy``.
* Set ``latency_factor`` and ``latency_threshold`` to control ``Alignment-based Latency Loss``.
* Run the following script for Stage-2 training. (The scripts can be also found in the [``shell_scripts``](https://github.com/ictnlp/NAST/tree/main/shell_scripts) directory.)

```bash
exp=your_exp_name
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/save_checkpoint
plugin_path=/path/to/NAST_plugins
wait_until=0
pretrain_model_path=/path/to/pretrained_model
latency_factor=0.0
latency_threshold=0.0

fairseq-train $data_dir \
    --user-dir ${plugin_path} \
    --fp16 \
    --finetune-from-model ${pretrain_model_path} \
    --save-dir ${checkpoint_dir} \
    --ddp-backend=legacy_ddp \
    --task translation_ctc_streaming \
    --criterion nat_loss_ngram_glat_simul --left-pad-source --glat-p 0.3:0.3@20k \
    --src-embedding-copy \
    --src-upsample-ratio 3 --plain-ctc --wait-until 0 --latency-factor ${latency_factor} --latency-threshold ${latency_threshold} \
    --arch nonautoregressive_streaming_transformer \
    --use-ngram --ngram-size 2 \
    --noise full_mask \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9,0.98)' \
    --lr 0.0003 --lr-scheduler inverse_sqrt \
    --stop-min-lr '1e-09' --warmup-updates 500 \
    --warmup-init-lr '1e-07' \
    --dropout 0.1 --weight-decay 0.01 \
    --decoder-learned-pos \
    --encoder-learned-pos \
    --log-format 'simple' --log-interval 10 \
    --fixed-validation-seed 7 \
    --max-tokens 1024 \
    --update-freq 64 \
    --save-interval-updates 500 \
    --max-update 10000
    
```

## Inference
We average the parameters of the last 5 checkpoints, empirically leading to a better performance.
```bash
checkpoint_dir=/path/to/save_checkpoint
plugin_path=/path/to/NAST_plugins
average_checkpoint_path=$checkpoint_dir/average_last_5.pt

python3 ${plugin_path}/scripts/average_checkpoints.py --inputs ${checkpoint_dir} \
                --num-update 5 --output ${average_checkpoint_path} \
```
We use the [``generate_streaming``](https://github.com/ictnlp/NAST/blob/main/NAST/scripts/generate_streaming.py) script to simulate streaming input scenarios and measure translation quality along with various latency metrics.
* Ensure that ``wait_until`` remains consistent with its usage during training.
* Run the following script for inference. (The scripts can be also found in the [``shell_scripts``](https://github.com/ictnlp/NAST/tree/main/shell_scripts) directory.)
```bash
average_checkpoint_path=/path/to/checkpoint
data_dir=/path/to/binarized_data
plugin_path=/path/to/NAST_plugins
wait_until=0

python ${plugin_path}/scripts/generate_streaming.py ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --src-upsample-ratio 3 --plain-ctc \
    --wait-until ${wait_until} \
    --model-overrides "{\"wait_until\":${wait_until},\"src_upsample_ratio\":3,\"src_embedding_copy\":True,\"plain_ctc\":True}" \
    --task translation_ctc_streaming \
    --path ${average_checkpoint_path} \
    --max-tokens 2048 --remove-bpe \
    --left-pad-source
```
If everything goes smoothly, you will get an output similar to the following.
```
Generate test with beam=5: BLEU4 = 29.92, 66.7/39.1/24.7/16.2 (BP=0.935, ratio=0.937, syslen=44626, reflen=47634)
CW score:  1.738971989129786
AP score:  0.6551425751117264
AL score:  4.017025168974748
DAL score:  5.888195282215187
```
Please note that **the BLEU score reported above is not directly comparable** to the scores in the literature. This is because the scores reported in the text simultaneous translation papers are calculated with the omission of letter capitalization.
To obtain a comparable BLEU score, Please make use of the [``multi-bleu.perl``](https://github.com/ictnlp/NAST/blob/main/shell_scripts/multi-bleu.perl) script.
```bash
gen=$1
ref=$2
cat $gen | grep -P "^D-" | sort -V |cut -f 2- > $gen.tok

perl multi-bleu.perl -lc $ref < $gen.tok
```
## Citing

Please kindly cite us if you find our papers or codes useful.

```
@inproceedings{
ma2023nonautoregressive,
title={Non-autoregressive Streaming Transformer for Simultaneous Translation},
author={Ma, Zhengrui and Zhang, Shaolei and Guo, Shoutao and Shao, Chenze and Zhang, Min and Feng, Yang
},
booktitle={Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
}
```
