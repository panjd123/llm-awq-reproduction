# llm-awq-reproduction

This repository contains the reproduction of the paper "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration".

## Table of Contents

- [Installation](#installation)

## Installation

To set up the environment, please use the following steps:

1. Clone the repository and the submodule:

```bash
git clone https://github.com/panjd123/llm-awq-reproduction.git
cd llm-awq-reproduction
git submodule update --init --recursive
cd llm-awq # this is the original implementation of AWQ, following instructions is as same as the original repository
```

2. Install Package
```
conda create -n awq python=3.10 -y
conda activate awq
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install efficient W4A16 (4-bit weight, 16-bit activation) CUDA kernel and optimized FP16 kernels (e.g. layernorm, positional encodings).
```
cd awq/kernels
python setup.py install
```

4. In order to run AWQ and TinyChat with VILA-1.5 model family, please install VILA:

```bash
git clone git@github.com:Efficient-Large-Model/VILA.git
cd VILA
pip install -e .
```
