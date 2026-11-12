
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

class Model(nn.Module):
    """
    Simple model that performs 1D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = 1, padding: int = 0):
        """
        Initializes the 1D Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to 1.
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 1D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).

        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied, shape (batch_size, in_channels, output_length).
        """
        return self.avg_pool(x)

# #####################
# Important:
# The above code is the reference implementation.
# The below code is what needs to be changed.
# #####################

# Define the custom CUDA kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__global__ void custom_kernel(const float* input, float* output, int size) {
    // custom kernel code
}

torch::Tensor custom_cuda(torch::Tensor input) {
    //custom code to invoke the kernel
}
"""

cpp_source = (
    "torch::Tensor custom_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softplus
custom_cuda_op = load_inline(
    name="custom_cuda_op",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["custom_cuda"],
    verbose=False,
    extra_cflags=[""],
    extra_ldflags=[""],
)

class ModelNew(nn.Module):
    """
    Optimized model using custom CUDA operator for Softplus activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.custom_cuda_op = custom_cuda_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        applies custom CUDA.
        """ 
        return self.custom_cuda_op.custom_cuda(x)
