MODEL=llama-2-7b

pip install . && python ./llm-awq/awq/entry.py --model_path /data/models/llama-2-7b \
--tasks wikitext \
--w_bit 4 --q_group_size 128 \
--load_quant /data/quant_cache/llama-2-7b-w4-g128-awq-v2.pt