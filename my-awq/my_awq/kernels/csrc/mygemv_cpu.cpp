#include <torch/extension.h>
#include "mygemv.h"

torch::Tensor mygemv_cpu(
    torch::Tensor _inputs,
    torch::Tensor _weight,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size) {
    throw std::runtime_error("mygemv_cpu is not implemented");
}