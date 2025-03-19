#include <torch/extension.h>

// Declaration of the CUDA function
torch::Tensor element_wise_add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const float alpha,
    const float beta);

// CPU implementation for reference
torch::Tensor element_wise_add_cpu(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const float alpha = 1.0,
    const float beta = 1.0) {
    
    return alpha * a + beta * b;
}

// Interface function that dispatches to either CPU or CUDA implementation
torch::Tensor element_wise_add(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const float alpha = 1.0,
    const float beta = 1.0) {
    
    if (a.device().is_cuda()) {
        return element_wise_add_cuda(a, b, alpha, beta);
    } else {
        return element_wise_add_cpu(a, b, alpha, beta);
    }
}

// Module definitions
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("element_wise_add", &element_wise_add, 
          "Element-wise addition with scaling factors (A * alpha + B * beta)",
          py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0);
}
