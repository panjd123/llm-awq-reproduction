#include "mygemv.h"

torch::Tensor mygemv_cuda(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
    int group_size) {
    throw std::runtime_error("mygemv_cuda is not implemented");
}