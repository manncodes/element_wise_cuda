import torch
import time
import element_wise_cuda

def benchmark(func, a, b, alpha=1.0, beta=1.0, name="", iterations=100):
    # Warmup
    for _ in range(10):
        result = func(a, b, alpha, beta)
    
    # Sync CUDA
    if a.is_cuda:
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        result = func(a, b, alpha, beta)
        if a.is_cuda:
            torch.cuda.synchronize()
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"{name} took {elapsed_time:.6f} seconds for {iterations} iterations (avg: {elapsed_time/iterations*1000:.4f} ms)")
    return result

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU only.")
    
    # Create test tensors
    size = 10_000_000
    a_cpu = torch.randn(size)
    b_cpu = torch.randn(size)
    
    # CPU PyTorch implementation
    native_cpu_result = benchmark(
        lambda a, b, alpha, beta: alpha * a + beta * b,
        a_cpu, b_cpu, 2.5, 1.5, name="PyTorch CPU")
    
    # Our CPU implementation
    our_cpu_result = benchmark(
        element_wise_cuda.element_wise_add,
        a_cpu, b_cpu, 2.5, 1.5, name="Our CPU impl")
    
    # Check for correctness on CPU
    assert torch.allclose(native_cpu_result, our_cpu_result)
    print("CPU implementations match!")
    
    if torch.cuda.is_available():
        # Move tensors to GPU
        a_cuda = a_cpu.cuda()
        b_cuda = b_cpu.cuda()
        
        # GPU PyTorch implementation
        native_cuda_result = benchmark(
            lambda a, b, alpha, beta: alpha * a + beta * b,
            a_cuda, b_cuda, 2.5, 1.5, name="PyTorch CUDA")
        
        # Our CUDA implementation
        our_cuda_result = benchmark(
            element_wise_cuda.element_wise_add,
            a_cuda, b_cuda, 2.5, 1.5, name="Our CUDA impl")
        
        # Check for correctness on GPU
        print(native_cuda_result)
        print(our_cuda_result)
        assert torch.allclose(native_cuda_result, our_cuda_result, atol=1e-6)
        print("CUDA implementations match!")

if __name__ == "__main__":
    main()
