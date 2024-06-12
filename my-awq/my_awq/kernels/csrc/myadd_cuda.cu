#include <cuda_fp16.h>
#include <torch/extension.h>
#include "myadd.h"

#define BLOCK_SIZE 256

__global__ void myadd_kernel(const half* a, const half* b, half* c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = __hadd(a[i], b[i]);
    }
}

torch::Tensor myadd_cuda(torch::Tensor a, torch::Tensor b) {
    assert(a.scalar_type() == at::kHalf);
    assert(b.scalar_type() == at::kHalf);
    assert(a.device().is_cuda());
    assert(b.device().is_cuda());
    assert(a.sizes() == b.sizes());
    const half* a_ptr = reinterpret_cast<const half*>(a.data_ptr<at::Half>());
    const half* b_ptr = reinterpret_cast<const half*>(b.data_ptr<at::Half>());
    torch::Tensor c = torch::empty_like(a);
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    const size_t n = a.numel();
    const size_t blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    myadd_kernel<<<blocks, BLOCK_SIZE>>>(a_ptr, b_ptr, c_ptr, n);
    return c;
}