exp=test_stage_2_1
data_dir=~/dataset/wmt15_deen_distill/data-bin-de-en
checkpoint_dir=./checkpoints/$exp
plugin_path=./NAST/NAST
wait_until=0
pretrain_model_path=~/exp_simul_CTC/checkpoints/simul_ctc_deen_distill_upsample_3_latency_0_glat/average_last_5.pt
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

#--keep-interval-updates 5 --keep-last-epochs 5 \
#--use-ngram --ngram-size 2 \
