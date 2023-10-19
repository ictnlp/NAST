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
