#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "my_gemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mygemv_cpu", &mygemv_cpu, "GEMV (CPU)");
    m.def("mygemv_cuda", &mygemv_cuda, "GEMV (CUDA)");
}
