#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "myadd.h"
#include "mygemv.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("myadd_cuda", &myadd_cuda, "Test cuda extension");
    m.def("mygemv_cpu", &mygemv_cpu, "GEMV (CPU)");
    m.def("mygemv_cuda", &mygemv_cuda, "GEMV (CUDA)");
}
