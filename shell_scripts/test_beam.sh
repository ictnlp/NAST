average_checkpoint_path=/data/mazhengrui/exp_simul_CTC/model.pt
data_dir=~/dataset/wmt15_deen_distill/data-bin-de-en
plugin_path=./NAST/NAST
wait_until=0

python ${plugin_path}/scripts/generate_streaming.py ${data_dir} \
    --user-dir ${plugin_path} \
    --gen-subset test \
    --src-upsample-ratio 3 \
    --wait-until ${wait_until} \
    --model-overrides "{\"wait_until\":${wait_until},\"src_upsample_ratio\":3,\"src_embedding_copy\":True,\"plain_ctc\":False}" \
    --task translation_ctc_streaming \
    --path ${average_checkpoint_path} \
    --max-tokens 2048 --remove-bpe \
    --left-pad-source > temp
