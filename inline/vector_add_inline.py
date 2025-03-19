import torch
from torch.utils.cpp_extension import load_inline

# Define the C++ source that includes our CUDA kernel
cpp_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition with a scaling factor
template <typename scalar_t>
__global__ void element_wise_add_kernel(
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ b,
    scalar_t* __restrict__ c,
    const scalar_t alpha,
    const scalar_t beta,
    const int size) {
    
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (index < size) {
        c[index] = alpha * a[index] + beta * b[index];
    }
}

// CUDA kernel launch wrapper
template <typename scalar_t>
void element_wise_add_cuda_impl(
    const torch::Tensor& a,
    const torch::Tensor& b,
    torch::Tensor& c,
    const scalar_t alpha,
    const scalar_t beta) {
    
    const int size = a.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    
    element_wise_add_kernel<scalar_t><<<blocks, threads>>>(
        a.data_ptr<scalar_t>(),
        b.data_ptr<scalar_t>(),
        c.data_ptr<scalar_t>(),
        alpha,
        beta,
        size);
}

// Interface function accessible from C++
torch::Tensor element_wise_add_cuda(
    const torch::Tensor& a,
    const torch::Tensor& b,
    const float alpha = 1.0,
    const float beta = 1.0) {
    
    // Input validation
    TORCH_CHECK(a.device().is_cuda(), "Input tensor 'a' must be a CUDA tensor");
    TORCH_CHECK(b.device().is_cuda(), "Input tensor 'b' must be a CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "Input tensors must have the same shape");
    
    // Create output tensor
    auto c = torch::empty_like(a);
    
    // Determine data type and call the appropriate implementation
    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "element_wise_add_cuda", ([&] {
        element_wise_add_cuda_impl<scalar_t>(
            a,
            b,
            c,
            static_cast<scalar_t>(alpha),
            static_cast<scalar_t>(beta));
    }));
    
    return c;
}

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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("element_wise_add", &element_wise_add, 
          "Element-wise addition with scaling factors (A * alpha + B * beta)",
          py::arg("a"), py::arg("b"), py::arg("alpha") = 1.0, py::arg("beta") = 1.0);
}
"""

# Load the extension
inline_extension = load_inline(
    name="element_wise_inline",
    cpp_sources=cpp_source,
    cuda_sources=[],  # CUDA code is included directly in the cpp_source
    functions=["element_wise_add"],
    verbose=True
)

def test_kernel():
    # Create sample tensors
    a_cpu = torch.randn(1000)
    b_cpu = torch.randn(1000)
    
    # Test CPU operation
    alpha, beta = 2.5, 1.5
    torch_result_cpu = alpha * a_cpu + beta * b_cpu
    custom_result_cpu = inline_extension.element_wise_add(a_cpu, b_cpu, alpha, beta)
    cpu_match = torch.allclose(torch_result_cpu, custom_result_cpu)
    print(f"CPU implementation correct: {cpu_match}")
    
    # Test CUDA operation if available
    if torch.cuda.is_available():
        a_cuda = a_cpu.cuda()
        b_cuda = b_cpu.cuda()
        
        torch_result_cuda = alpha * a_cuda + beta * b_cuda
        custom_result_cuda = inline_extension.element_wise_add(a_cuda, b_cuda, alpha, beta)
        cuda_match = torch.allclose(torch_result_cuda, custom_result_cuda)
        print(f"CUDA implementation correct: {cuda_match}")
    else:
        print("CUDA not available, skipping GPU test")

if __name__ == "__main__":
    test_kernel()