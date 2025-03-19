# Element-wise CUDA Operations for PyTorch

This repository provides a simple CUDA kernel extension for PyTorch that performs element-wise operations on tensors. It's designed to be used as a reference implementation for creating custom CUDA operations in PyTorch.

## Features

- Element-wise addition with scaling factors: `C = alpha * A + beta * B`
- Supports both CPU and CUDA tensors
- Automatically dispatches to the appropriate implementation based on input tensor device
- Simple benchmarking tool included

## Installation

### Prerequisites

- CUDA toolkit (compatible with your PyTorch installation)
- PyTorch >=1.7.0
- A C++ compiler compatible with your PyTorch and CUDA versions

### Installation Steps

1. Clone this repository:
   ```bash
   git clone https://github.com/manncodes/element_wise_cuda.git
   cd element_wise_cuda
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Usage

```python
import torch
import element_wise_cuda

# Create some tensors
a = torch.randn(1000, 1000).cuda()
b = torch.randn(1000, 1000).cuda()

# Use our custom CUDA kernel
result = element_wise_cuda.element_wise_add(a, b, alpha=2.0, beta=3.0)

# This is equivalent to the following PyTorch operation
pytorch_result = 2.0 * a + 3.0 * b

# Verify correctness
assert torch.allclose(result, pytorch_result)
```

## Benchmarking

You can run the included benchmark script to compare performance:

```bash
python example.py
```

## Integrating with Your ml-sys Repository

To integrate this CUDA extension into your ml-sys repository:

1. Copy the `element_wise_cuda` directory to your repository
2. Add `element_wise_cuda` as a dependency in your project's setup.py or requirements.txt
3. Import and use the functions as shown in the example

You can modify the kernel for different operations or expand it to include more complex operations as needed.

## License

MIT
