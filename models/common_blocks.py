import math
import numbers
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from mmcv.utils import get_logger
import logging

def get_root_logger(log_file=None, log_level=logging.INFO):
    """Use ``get_logger`` method in mmcv to get the root logger.
    The logger will be initialized if it has not been initialized. By default a
    StreamHandler will be added. If ``log_file`` is specified, a FileHandler
    will also be added. The name of the root logger is the top-level package
    name, e.g., "mmaction".
    Args:
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the root logger.
        log_level (int): The root logger level. Note that only the process of
            rank 0 is affected, while other processes will set the level to
            "Error" and be silent most of the time.
    Returns:
        :obj:`logging.Logger`: The root logger.
    """
    return get_logger(__name__.split('.')[0], log_file, log_level)

def gaussian_2d_kernel(kernel_size = 3,sigma = 0):
    kernel = np.zeros([kernel_size,kernel_size])
    center = kernel_size//2
    
    if sigma == 0:
        sigma = ((kernel_size-1)*0.5 - 1)*0.3 + 0.8
    
    s = 2*(sigma**2)
    sum_val = 0
    for i in range(0,kernel_size):
        for j in range(0,kernel_size):
            x = i-center
            y = j-center
            kernel[i,j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernel[i,j]
            #/(np.pi * s)
    sum_val = 1/sum_val
    return kernel*sum_val

class GaussianFliterModule(nn.Module):
    """ Gaussian Fliter Module """

    def __init__(self, in_channels=3, kernel_size=5):
        super().__init__()
        if kernel_size % 2 == 0:
            raise NotImplementedError("Requires odd kernel size for GaussianFliterModule")
        self.channels = in_channels
        self.pad = int((kernel_size - 1) // 2)
        kernel = gaussian_2d_kernel(kernel_size=kernel_size)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)    # (H, W) -> (1, 1, H, W)
        kernel = kernel.expand((int(in_channels), 1, kernel_size, kernel_size))
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
    
    def __call__(self, x):
        x = F.conv2d(x, self.weight, padding=self.pad, groups=self.channels)
        return x

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [torch.arange(size, dtype=torch.float32) for size in kernel_size])
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        self.conv = nn.Conv2d(1,
                              1,
                              kernel_size=kernel_size,
                              padding=0,
                              bias=False)
        with torch.no_grad():
            self.conv.weight.data = kernel

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input)
        # return self.conv(input, weight=self.weight, groups=self.groups)
