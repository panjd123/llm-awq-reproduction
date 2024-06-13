# llm-awq-reproduction

This repository contains the end2end reproduction of the paper "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration".

## Table of Contents

- [Installation](#installation)
- [Reproduce the Results on Llama-2-7b](#reproduce-the-results-on-llama-2-7b)

## Introduction

Based on [llm-awq](https://github.com/mit-han-lab/llm-awq), commit `ca11f3`.

But modified the following to make it work:

- Add `config.use_cache = False` to avoid oom.
- Manually implement ppl evaluation for wikitext.

Why commit `ca11f3`:

- The nearest major commit before `5f06db`, which introduces a much more complex CUDA kernel.

## Installation

To set up the environment, please use the following steps:

1. Clone the repository and the submodule:

```bash
git clone https://github.com/panjd123/llm-awq-reproduction.git
cd llm-awq-reproduction
git submodule update --init --recursive
```

2. Install Original AWQ package:

```bash
cd llm-awq # this is the original implementation of AWQ
conda create -n awq python=3.10 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).

```bash
cd awq/kernels
python setup.py install
```

4. Install our reproduction package:

```bash
cd ../../../ # go back to the root directory of this repository
cd my-awq
pip install -e .
```

5. Install our W4A16 CUDA kernel (optional, we use awq's official kernel by default):

```bash
cd my_awq/kernels
python setup.py install
```

## Reproduce the Results on Llama-2-7b

### If you can access huggingface.co:

```bash
mkdir -p /data/my_awq_cache
mkdir -p /data/my_quant_cache

python -m my_awq.entry --model_path meta-llama/Llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --calib_dataset_path mit-han-lab/pile-val-backup --dump_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path meta-llama/Llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend fake \
    --run_eval --wikitext_path wikitext

python -m my_awq.entry --model_path meta-llama/Llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --dump_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path meta-llama/Llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --run_eval --wikitext_path wikitext
    
python -m my_awq.entry --model_path meta-llama/Llama-2-7b \
    --run_eval --wikitext_path wikitext
```

### Otherwise:

Download the models and datasets to the following paths:

- https://huggingface.co/meta-llama/Llama-2-7b to `/data/models/llama-2-7b`
- https://huggingface.co/datasets/mit-han-lab/pile-val-backup to `/data/datasets/pile-val-backup`
- https://huggingface.co/datasets/Salesforce/wikitext to `/data/datasets/wikitext`

```bash
mkdir -p /data/my_awq_cache
mkdir -p /data/my_quant_cache

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --run_awq --calib_dataset_path /data/datasets/pile-val-backup --dump_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend fake \
    --run_eval --wikitext_path /data/datasets/wikitext

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_awq /data/my_awq_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --dump_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt

python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --w_bit 4 --q_group_size 128 \
    --load_quant /data/my_quant_cache/llama-2-7b-w4-g128.pt --q_backend real \
    --run_eval --wikitext_path /data/datasets/wikitext
    
python -m my_awq.entry --model_path /data/models/llama-2-7b \
    --run_eval --wikitext_path /data/datasets/wikitext
```