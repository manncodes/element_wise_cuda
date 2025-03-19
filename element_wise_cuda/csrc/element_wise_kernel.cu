#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for element-wise addition with a scaling factor
// C = alpha * A + beta * B
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
