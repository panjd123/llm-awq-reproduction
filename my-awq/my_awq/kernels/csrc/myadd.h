#pragma once
#include <torch/extension.h>

torch::Tensor myadd_cuda(const torch::Tensor& a, const torch::Tensor& b);