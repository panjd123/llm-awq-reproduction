#pragma once

#include <torch/extension.h>

torch::Tensor mygemv_cuda(
    torch::Tensor _inputs,
    torch::Tensor _weight,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size);

torch::Tensor mygemv_cpu(
    torch::Tensor _inputs,
    torch::Tensor _weight,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size);