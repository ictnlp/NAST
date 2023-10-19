exp=your_exp_name
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/save_checkpoint
plugin_path=/path/to/NAST_plugins
wait_until=0
pretrain_model_path=/path/to/pretrained_model
latency_factor=1.0
latency_threshold=1.0

nohup fairseq-train $data_dir \
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
    --max-update 10000 > logs/$exp.txt &

