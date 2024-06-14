mkdir -p /data/my_awq_cache
mkdir -p /data/my_quant_cache

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --calib_dataset_path mit-han-lab/pile-val-backup --dump_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend fake \
    --run_eval --wikitext_path wikitext

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --dump_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --run_eval --wikitext_path wikitext
    
python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --run_eval --wikitext_path wikitext