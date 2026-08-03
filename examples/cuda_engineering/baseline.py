
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define the custom CUDA kernel for Softplus activation
softplus_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cmath>

__device__ __forceinline__ float softplus(float x) {
    // Numerically stable softplus: log(1 + exp(x)) 
    // For large x, exp(x) dominates, so we use x
    // For small x, we can use log(1 + exp(x))
    if (x > 30.0f) {
        return x;
    } else if (x < -30.0f) {
        return expf(x);
    } else {
        return logf(1.0f + expf(x));
    }
}

__global__ void softplus_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = softplus(input[idx]);
    }
}

torch::Tensor softplus_cuda(torch::Tensor input) {
    auto size = input.numel();
    auto output = torch::zeros_like(input);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    softplus_kernel<<<num_blocks, block_size>>>(
        input.data_ptr<float>(), 
        output.data_ptr<float>(), 
        size
    );

    return output;
}
"""

softplus_cpp_source = (
    "torch::Tensor softplus_cuda(torch::Tensor input);"
)

# Compile the inline CUDA code for Softplus
softplus_op = load_inline(
    name="softplus_op",
    cpp_sources=softplus_cpp_source,
    cuda_sources=softplus_source,
    functions=["softplus_cuda"],
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
        self.softplus_op = softplus_op
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        # Ensure input is on GPU and FP32
        if not x.is_cuda:
            x = x.cuda()
        if x.dtype != torch.float32:
            x = x.float()
            
        return self.softplus_op.softplus_cuda(x)
