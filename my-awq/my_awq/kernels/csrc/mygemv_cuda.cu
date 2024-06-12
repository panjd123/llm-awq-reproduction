#include <cuda_fp16.h>
#include <torch/extension.h>
#include <cstdio>
#include "mygemv.h"

#define WARP_SIZE 32
#define INPUTS_PACK_FACTOR 8  // float4 <==> 8 float16
#define WEIGHT_PACK_FACTOR 8  // int32 <==> 8 int4

#if INPUTS_PACK_FACTOR != WEIGHT_PACK_FACTOR
#error "INPUTS_PACK_FACTOR must be equal to WEIGHT_PACK_FACTOR"
#endif

#define PACK_FACTOR 8  // works when INPUTS_PACK_FACTOR == WEIGHT_PACK_FACTOR
// our design is based on the assumption that INPUTS_PACK_FACTOR == WEIGHT_PACK_FACTOR == 8

#define GROUP_SIZE 128

#define N_CELL 4

__device__ int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}

__device__ int align_up(int a, int b) {
    return ((a + b - 1) / b) * b;
}

__device__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

/*
G = IC / group_size

_inputs: [N, IC] float16
_weight: [OC, IC / 8] int32 --> [OC, IC] int4
_scales: [OC, G] float16
_zeros: [OC, G / 8] int32 --> [OC, G] int4

_outputs: [N, OC] float16
*/

__global__ void _gemv_kernel(
    const half* inputs,
    const uint32_t* weight,
    const half* scales,
    const uint32_t* zeros,
    half* outputs,
    const int IC,
    const int OC) {
    const int n_group = IC / GROUP_SIZE;
    const int n_group_packed = ceil_div(n_group, PACK_FACTOR);

    const int target_i = blockIdx.x;
    const int target_j = blockIdx.y * N_CELL + threadIdx.y;

    const int weights_width = IC / PACK_FACTOR;
    const int zeros_width = ceil_div(n_group, PACK_FACTOR);
    const int scales_width = align_up(n_group, PACK_FACTOR);

    float4* inputs_float4 = (float4*)(inputs + target_i * IC);

    float sum = 0;
    for (int pack_group_id = 0; pack_group_id < n_group_packed; pack_group_id++) {  // a warp handle 8 group in each iteration
#define THREAD_PER_GROUP (WARP_SIZE / PACK_FACTOR)
        const int sub_group_id = threadIdx.x / THREAD_PER_GROUP;  // group_id in packed group
        // const int id_in_group = threadIdx.x % THREAD_PER_GROUP;   // id in group
        const int group_id = pack_group_id * PACK_FACTOR + sub_group_id;
        // 0 1 2 3 | 4 5 6 7 | 8 9 10 11 | 12 13 14 15 | 16 17 18 19 | 20 21 22 23 | 24 25 26 27 | 28 29 30 31

        const int cur_packed_element_start_id = (pack_group_id * WARP_SIZE + threadIdx.x) * 4;
        // per thread handle 32 elements
        // now is the (pack_group_id * WARP_SIZE + threadIdx.x)th thread in a row
        // tid = pack_group_id * WARP_SIZE + threadIdx.x
        // now we should handle from 32*tid to 32*tid+31
        // means from 4*tid to 4*tid+3 in packed element (float32 inputs, int32 weights)

        uint32_t packed_zero = zeros[target_j * zeros_width + pack_group_id];
        float zero = static_cast<float>((packed_zero >> (sub_group_id * 4)) & 0xF);  // 4 bit
        float scale = __half2float(scales[target_j * scales_width + group_id]);
        uint32_t packed_weights[4];
        *((int4*)packed_weights) = *((int4*)(weight + target_j * weights_width + cur_packed_element_start_id));
        // load in 1 loop

        for (int i = 0; i < 4; i++) {
            uint32_t cur_packed_weight = packed_weights[i];
            // int32* weight_int32 = weight + weight + target_j * weights_width;
            // uint32_t cur_packed_weight = weight_int32[cur_packed_element_start_id + i]

            half packed_inputs[PACK_FACTOR];
            if (cur_packed_element_start_id + i < weights_width) {
                *((float4*)packed_inputs) = inputs_float4[cur_packed_element_start_id + i];  // load in 4 loops
                for (int j = 0; j < PACK_FACTOR; j++) {
                    float cur_weight = static_cast<float>(cur_packed_weight & 0xF);
                    float dequantized_weight = (cur_weight - zero) * scale;
                    float input = __half2float(packed_inputs[j]);
                    sum += dequantized_weight * input;
                    cur_packed_weight >>= 4;
                }
            }
        }
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x == 0) {
        outputs[target_i * OC + target_j] = __float2half(sum);
    }
}

/*
G = IC / group_size

_inputs: [N, IC] float16
_weight: [OC, IC / 8] int32 --> [OC, IC] int4
_scales: [OC, G] float16
_zeros: [OC, G / 8] int32 --> [OC, G] int4

_outputs: [N, OC] float16
*/

torch::Tensor mygemv_cuda(
    torch::Tensor _inputs,
    torch::Tensor _weight,
    torch::Tensor _scales,
    torch::Tensor _zeros,
    int group_size) {
    const int N = _inputs.size(0);
    const int IC = _inputs.size(1);
    const int OC = _weight.size(0);
    assert(_inputs.scalar_type() == at::kHalf);
    assert(_weight.scalar_type() == at::kInt);
    assert(_scales.scalar_type() == at::kHalf);
    assert(_zeros.scalar_type() == at::kInt);
    auto options = torch::TensorOptions().dtype(_inputs.dtype()).device(_inputs.device());
    torch::Tensor _outputs = torch::empty({N, OC}, options);
    // each block handles 4 cell in the output matrix
    // 1 warp handle 1 cell in the output matrix
    assert(IC % PACK_FACTOR == 0);
    assert(OC % N_CELL == 0);
    dim3 blocks(N, OC / N_CELL);
    dim3 threads(WARP_SIZE, N_CELL);
    const half* inputs = reinterpret_cast<half*>(_inputs.data_ptr<at::Half>());
    const uint32_t* weight = reinterpret_cast<uint32_t*>(_weight.data_ptr<int>());
    const half* scales = reinterpret_cast<half*>(_scales.data_ptr<at::Half>());
    const uint32_t* zeros = reinterpret_cast<uint32_t*>(_zeros.data_ptr<int>());
    half* outputs = reinterpret_cast<half*>(_outputs.data_ptr<at::Half>());
    _gemv_kernel<<<blocks, threads>>>(
        inputs,
        weight,
        scales,
        zeros,
        outputs,
        IC,
        OC);
    return _outputs;
}