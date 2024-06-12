#pragma once
#include <torch/extension.h>

torch::Tensor myadd_cuda(torch::Tensor a, torch::Tensor b);