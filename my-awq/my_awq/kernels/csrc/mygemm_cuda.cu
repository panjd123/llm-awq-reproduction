#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstdio>
#include "mygemm.h"

/*
G = IC / group_size

_inputs: [N, IC] float16
_weight: [OC, IC / 8] int32 --> [OC, IC] int4
_scales: [OC, G] float16
_zeros: [OC, G / 8] int32 --> [OC, G] int4

_outputs: [N, OC] float16
*/

torch::Tensor mygemm_cuda(
    torch::Tensor _inputs,
    torch::Tensor _weight,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size) {
    throw std::runtime_error("mygemm_cuda is not implemented");
}