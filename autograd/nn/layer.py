from typing import Union, Tuple

import numpy as np

import autograd
from autograd import Tensor
from module import Module
from parameter import Parameter
import functional


class Conv2d(Module):
    """
    Implementation of a 2d convolution in autograd
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int]] = None, bias: bool = True) -> None:
        # Call super constructor
        super(Conv2d, self).__init__()
        # Convert kernel size to a tuple
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # Convert padding to a tuple and save it
        if padding is not None:
            self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        else:
            self.padding = padding
        # Init weights
        self.weight = Parameter(out_channels, in_channels, kernel_size[0], kernel_size[1])
        self.bias = Parameter(out_channels) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor of shape (batch size, in channels, height, width)
        :return: (Tensor) Output tensor
        """
        if self.padding is not None:
            input = autograd.pad_2d(input, pad_width=(self.padding, self.padding))
        return functional.conv_2d(input=input, kernel=self.weight, bias=self.bias)


class MaxPool2d(Module):
    """
    Implementation of a 2d max-pooling module in autograd
    """

    def __init__(self, kernel_size: Union[int, Tuple[int, int]]) -> None:
        # Call super constructor
        super(MaxPool2d, self).__init__()
        # Convert kernel size to a tuple
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # Check kernel size to be even
        assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0, 'Kernel size must be even.'
        # Save kernel size
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor of shape (batch size, channels, height, width)
        :return: (Tensor) Output tensor
        """
        return functional.max_pool_2d(tensor=input, kernel_size=self.kernel_size)


class AvgPool2d(Module):
    """
    Implementation of a 2d avg-pooling module in autograd
    """

    def __init__(self, kernel_size: Union[int, Tuple[int, int]]) -> None:
        # Call super constructor
        super(AvgPool2d, self).__init__()
        # Convert kernel size to a tuple
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        # Check kernel size to be even
        assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0, 'Kernel size must be even.'
        # Save kernel size
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor of shape (batch size, channels, height, width)
        :return: (Tensor) Output tensor
        """
        return functional.avg_pool_2d(tensor=input, kernel_size=self.kernel_size)


class BatchNorm1d(Module):
    """
    Class implements a batch normalization layer
    """

    def __init__(self, num_channels: int, eps: float = 1e-05, momentum=0.1, affine: bool = True,
                 track_running_stats: bool = True) -> None:
        # Call super constructor
        super(BatchNorm1d, self).__init__()
        # Save parameter
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        # Init learnable parameter if utilized
        self.gamma = Parameter(data=np.ones(num_channels)) if affine else None
        self.beta = Parameter(data=np.zeros(num_channels)) if affine else None
        # Init running mean and std if needed
        self.running_mean = Tensor(0.0) if self.track_running_stats else None
        self.running_std = Tensor(1.0) if self.track_running_stats else None

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward method
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        if self.train_mode and self.track_running_stats:
            output, mean, std = functional.batch_norm_2d(input, gamma=self.gamma, beta=self.beta, eps=self.eps,
                                                         return_mean_and_std=True)
            # Apply running mean and std
            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * mean.data
            self.running_std.data = self.momentum * self.running_std.data + (1 - self.momentum) * std.data
            return output
        else:
            return functional.batch_norm_2d(input, gamma=self.gamma, beta=self.beta, eps=self.eps,
                                            mean=self.running_mean, std=self.running_std, return_mean_and_std=False)


class UpsamplingNearest2d(Module):
    """
    Implementation of a nearest neighbour upsampling module.
    """

    def __init__(self, scale_factor: int = 2) -> None:
        """
        Constructor
        :param scale_factor: (int) Scaling factor
        """
        # Call super constructor
        super(UpsamplingNearest2d, self).__init__()
        # Check parameter
        assert scale_factor > 0, 'Scale factor must be greater than zero.'
        # Save parameter
        self.scale_factor = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Upscaled output tensor
        """
        return functional.upsampling_nearest_2d(input=input, scale_factor=self.scale_factor)


class UpsamplingNearest1d(Module):
    """
    Implementation of a nearest neighbour upsampling module.
    """

    def __init__(self, scale_factor: int = 2) -> None:
        """
        Constructor
        :param scale_factor: (int) Scaling factor
        """
        # Call super constructor
        super(UpsamplingNearest1d, self).__init__()
        # Check parameter
        assert scale_factor > 0, 'Scale factor must be greater than zero.'
        # Save parameter
        self.scale_factor = scale_factor

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Upscaled output tensor
        """
        return functional.upsampling_nearest_1d(input=input, scale_factor=self.scale_factor)


class Conv1d(Module):
    """
    Implementation of a 1d convolution layer.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, padding: int = None,
                 bias: bool = True) -> None:
        # Call super constructor
        super(Conv1d, self).__init__()
        # Save padding parameter
        self.padding = padding
        # Init weight
        self.weight = Parameter(out_channels, in_channels, kernel_size)
        # Init bias
        self.bias = Parameter(out_channels) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        # Perform padding if utilized
        if self.padding is not None:
            input = autograd.pad_1d(input, pad_width=(self.padding, self.padding))
        # Perform convolution
        return functional.conv_1d(input=input, kernel=self.weight, bias=self.bias)


class Linear(Module):
    """
    Implementation of a linear layer.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        """
        Constructor
        :param in_features: (int) Number of input channels
        :param out_features: (int) Number of output channels
        :param bias: (bool) True if bias should be utilized
        """
        # Call super constructor
        super(Linear, self).__init__()
        # Init weight
        self.weight = Parameter(out_features, in_features)
        # Init bias
        self.bias = Parameter(out_features) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor of shape (batch size, *, input features), * various and optional dimension
        :return: (Tensor) Output tensor of shape (batch size, *, output features), * various and optional dimension
        """
        # Add dimension to perform matmul batch-wise
        return functional.linear(input=input, weight=self.weight, bias=self.bias)


class Dropout(Module):
    """
    Class implements a dropout layer
    """

    def __init__(self, p: float = 0.2) -> None:
        """
        Constructor
        :param p: (float) Probability that a activation element is set to zero
        """
        # Call super constructor
        super(Dropout, self).__init__()
        # Check argument
        assert 0.0 <= p <= 1.0, 'Parameter p must be in the range of [0, 1].'
        # Save argument
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        if self.train_mode:
            return functional.dropout(input, p=self.p)
        else:
            return input


class BatchNorm1d(Module):
    """
    Class implements a batch normalization layer
    """

    def __init__(self, num_channels: int, eps: float = 1e-05, momentum=0.1, affine: bool = True,
                 track_running_stats: bool = True) -> None:
        # Call super constructor
        super(BatchNorm1d, self).__init__()
        # Save parameter
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        # Init learnable parameter if utilized
        self.gamma = Parameter(data=np.ones(num_channels)) if affine else None
        self.beta = Parameter(data=np.zeros(num_channels)) if affine else None
        # Init running mean and std if needed
        self.running_mean = Tensor(0.0) if self.track_running_stats else None
        self.running_std = Tensor(1.0) if self.track_running_stats else None

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward method
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        if self.train_mode and self.track_running_stats:
            output, mean, std = functional.batch_norm_1d(input, gamma=self.gamma, beta=self.beta, eps=self.eps,
                                                         return_mean_and_std=True)
            # Apply running mean and std
            self.running_mean.data = self.momentum * self.running_mean.data + (1 - self.momentum) * mean.data
            self.running_std.data = self.momentum * self.running_std.data + (1 - self.momentum) * std.data
            return output
        else:
            return functional.batch_norm_1d(input, gamma=self.gamma, beta=self.beta, eps=self.eps,
                                            mean=self.running_mean, std=self.running_std, return_mean_and_std=False)


class MaxPool1d(Module):
    """
    Class implements a 1d max-pooling operation module
    """

    def __init__(self, kernel_size: int = 2) -> None:
        """
        Constructor
        :param kernel_size: (int) Even kernel size
        """
        # Call super constructor
        super(MaxPool1d, self).__init__()
        # Check kernel size to be even
        assert kernel_size % 2 == 0, 'Kernel size must be even.'
        # Save kernel size
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return functional.max_pool_1d(tensor=input, kernel_size=self.kernel_size)


class AvgPool1d(Module):
    """
    Class implements a 1d max-pooling operation module
    """

    def __init__(self, kernel_size: int = 2) -> None:
        """
        Constructor
        :param kernel_size: (int) Even kernel size
        """
        # Call super constructor
        super(AvgPool1d, self).__init__()
        # Check kernel size to be even
        assert kernel_size % 2 == 0, 'Kernel size must be even.'
        # Save kernel size
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return functional.avg_pool_1d(tensor=input, kernel_size=self.kernel_size)
