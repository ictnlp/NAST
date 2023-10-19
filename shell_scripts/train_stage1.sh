exp=your_exp_name
data_dir=/path/to/binarized_data
checkpoint_dir=/path/to/save_checkpoint
plugin_path=/path/to/NAST_plugins
wait_until=0

nohup fairseq-train $data_dir \
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
    --max-update 300000 > logs/$exp.txt &