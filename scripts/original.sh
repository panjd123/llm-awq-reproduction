MODEL=llama-2-7b

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /data/models/$MODEL \
--w_bit 4 --q_group_size 128 \
--run_awq --dump_awq /data/awq_cache/$MODEL-w4-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path /data/models/$MODEL \
--tasks wikitext \
--w_bit 4 --q_group_size 128 \
--load_awq /data/awq_cache/$MODEL-w4-g128.pt \
--q_backend fake

# generate real quantized weights (w4)
python -m awq.entry --model_path /data/models/$MODEL \
--w_bit 4 --q_group_size 128 \
--load_awq /data/awq_cache/$MODEL-w4-g128.pt \
--q_backend real --dump_quant /data/quant_cache/$MODEL-w4-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path /data/models/$MODEL \
--tasks wikitext \
--w_bit 4 --q_group_size 128 \
--load_quant /data/quant_cache/$MODEL-w4-g128-awq-v2.pt