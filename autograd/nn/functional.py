from typing import List, Union, Tuple, Optional

import autograd
from autograd import Tensor
from autograd.tensor import Dependency

import numpy as np


def _conv_2d_core(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Implements convolution operation in numpy
    :param input: (np.ndarray) Input array of shape (batch size, input channels, height, width)
    :param kernel: (np.ndarray) Kernel matrix of shape (output channels, input channels, kernel size, kernel size)
    :return: (np.ndarray) Output array
    """
    # Change axis
    input = input.transpose((0, 2, 3, 1))
    kernel = kernel.transpose((2, 3, 1, 0))
    # Reshape
    input = np.lib.stride_tricks.as_strided(input, (input.shape[0],
                                                    input.shape[1] - kernel.shape[0] + 1,
                                                    input.shape[2] - kernel.shape[1] + 1,
                                                    kernel.shape[0],
                                                    kernel.shape[1],
                                                    input.shape[3]),
                                            input.strides[:3] + input.strides[1:])
    # Perform convolution and transpose
    return np.tensordot(input, kernel, axes=3).transpose((0, 3, 1, 2))


def conv_2d(input: Tensor, kernel: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    This function implements a 2d convolution (cross-correlation) in autograd
    :param input: (Tensor) Input tensor of shape (batch size, input channels, height, width)
    :param kernel: (Tensor) Kernel of shape (output channels, input channels, kernel size, kernel size)
    :param bias: (Tensor) Bias tensor of shape (output channels)
    :return: (Tensor) Output tensor (batch size, output channels, height - kernel size + 1, width - kernel size + 1)
    """
    # Check dimensions of parameters
    assert input.data.ndim == 4, 'Input tensor must have four dimensions.'
    assert kernel.data.ndim == 4, 'Kernel tensor must have four dimensions.'
    assert input.shape[1] == kernel.shape[1], 'Kernel features and input features must match.'
    # Perform forward pass
    output = _conv_2d_core(input.data, kernel.data)
    # Check if gradient is required
    requires_grad = input.requires_grad or kernel.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if input.requires_grad:
        # Make gradient function
        def grad_conv2d_input(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Pad gradient for convolution
            grad_padded = np.pad(grad.data, ((0, 0), (0, 0),
                                             (kernel.shape[2] - 1, kernel.shape[2] - 1),
                                             (kernel.shape[3] - 1, kernel.shape[3] - 1)), 'constant',
                                 constant_values=(1))
            # Convolve gradient with transposed kernel to get new gradient
            grad = _conv_2d_core(grad_padded, kernel.data.transpose((1, 0, 2, 3)))
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=input, grad_fn=grad_conv2d_input))

    if kernel.requires_grad:
        # Make gradient function
        def grad_conv2d_kernel(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Backward convolution
            grad = _conv_2d_core(input.data.transpose((1, 0, 2, 3)), grad.transpose((1, 0, 2, 3))).transpose(
                (1, 0, 2, 3))
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=kernel, grad_fn=grad_conv2d_kernel))

    output_conv = Tensor(data=output, dependencies=dependencies, requires_grad=requires_grad)
    # Apply bias if utilized
    if bias is None:
        return output_conv
    assert bias.data.ndim == 1, 'Bias tensor must have three dimensions.'
    # Reshape bias tensor to match output of matrix multiplication
    bias_batched = np.expand_dims(np.expand_dims(np.expand_dims(bias.data, axis=0), axis=-1), axis=-1)
    # Perform addition
    output_bias_add = output_conv.data + bias_batched
    # Check if gradient is required
    requires_grad = output_conv.requires_grad or bias.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if output_conv.requires_grad:
        # Make gradient function
        def grad_conv_output(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_conv, grad_fn=grad_conv_output))

    if bias.requires_grad:
        # Make gradient function
        def grad_conv_bias(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_conv, grad_fn=grad_conv_bias))

    return Tensor(data=output_bias_add, dependencies=dependencies, requires_grad=requires_grad)


def max_pool_2d(tensor: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """
    This function implements the 2d max pooling operation in autograd
    :param tensor: (Tensor) Input tensor of shape (batch size, channels, height, width)
    :param kernel_size: (Tuple[int]) Kernel size of the pooling operation.
    :return: (Tensor) Output tensor
    """
    # Check input dimensions
    assert tensor.data.ndim == 4, 'Input tensor must have four dimensions (batch size, channels, features).'
    # Check kernel size
    assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0, 'Kernel size must be odd!'
    # Check tensor shape
    assert tensor.shape[2] % 2 == 0 and tensor.shape[3] % 2 == 0, 'Tensor height and width must be odd!'
    # Get shape
    batch_size, channels, height, width = tensor.shape
    # Calc factors
    height_factor = height // kernel_size[0]
    width_factor = width // kernel_size[1]
    # Perform max pooling
    input_reshaped = tensor.data[:, :, :height_factor * kernel_size[0], :width_factor * kernel_size[1]] \
        .reshape(batch_size, channels, height_factor, kernel_size[0], width_factor, kernel_size[1])
    output = input_reshaped.max(axis=(3, 5))
    # Get indexes of max values
    indexes = \
        (tensor.data == np.repeat(np.repeat(output, kernel_size[0], axis=2), kernel_size[1], axis=3)).astype(float)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_max_pool_2d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Repeat gradient by kernel size
            unpooled_grad = np.repeat(np.repeat(grad.data, kernel_size[0], axis=2), kernel_size[1], axis=3)
            # Mask out not used elements
            grad = unpooled_grad * indexes
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_max_pool_2d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def avg_pool_2d(tensor: Tensor, kernel_size: Tuple[int, int]) -> Tensor:
    """
    This function implements the 2d average pooling operation in autograd
    :param tensor: (Tensor) Input tensor of shape (batch size, channels, height, width)
    :param kernel_size: (Tuple[int]) Kernel size of the pooling operation.
    :return: (Tensor) Output tensor
    """
    # Check input dimensions
    assert tensor.data.ndim == 4, 'Input tensor must have four dimensions (batch size, channels, features).'
    # Check kernel size
    assert kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0, 'Kernel size must be odd!'
    # Check tensor shape
    assert tensor.shape[2] % 2 == 0 and tensor.shape[3] % 2 == 0, 'Tensor height and width must be odd!'
    # Get shape
    batch_size, channels, height, width = tensor.shape
    # Calc factors
    height_factor = height // kernel_size[0]
    width_factor = width // kernel_size[1]
    # Perform max pooling
    input_reshaped = tensor.data[:, :, :height_factor * kernel_size[0], :width_factor * kernel_size[1]] \
        .reshape(batch_size, channels, height_factor, kernel_size[0], width_factor, kernel_size[1])
    output = input_reshaped.mean(axis=(3, 5))
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_avg_pool_2d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Repeat gradient by kernel size
            unpooled_grad = np.repeat(np.repeat(grad.data, kernel_size[0], axis=2), kernel_size[1], axis=3)
            # Mask out not used elements
            grad = (1 / (kernel_size[0] * kernel_size[1])) * unpooled_grad
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_avg_pool_2d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def batch_norm_1d(tensor: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None,
                  mean: Optional[Tensor] = None, std: Optional[Tensor] = None,
                  eps: float = 1e-05, running_mean: Optional[Tensor] = None, running_std: Optional[Tensor] = None,
                  momentum: float = None) -> Union[Tensor, Tuple[Tensor, Tensor, Tensor]]:
    """
    Function implements the 1D batch normalization operation
    :param tensor: (Tensor) Input tensor
    :param gamma: (Tensor) Learnable gamma factor which is multiplied to the output
    :param beta: (Tensor) Learnable beta factor which is added to the output
    :param mean: (Tensor) Mean for normalization
    :param std: (Tensor) Variance for normalization
    :param eps: (float) Constant for numerical stability
    :param running_mean: (Optional[Tensor]) Running mean
    :param running_std: (Optional[Tensor]) Running std
    :param momentum: (Optional[float]) Momentum of running statistics
    :return: (Tensor, Tuple[Tensor, Tensor, Tensor]) Output tensor and optional mean and std tensor
    """
    # Compute mean and std if not given
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    # If running statistics are given perform update
    if running_mean is not None and running_std is not None and momentum is not None:
        running_mean.data = momentum * running_mean.data + (1. - momentum) * mean.data
        running_std.data = momentum * running_std.data + (1. - momentum) * std.data
        mean.data = running_mean.data
        std.data = running_std.data
    # Apply normalization
    output = (tensor - mean) / (std + eps)
    # Apply learnable parameters if given
    if gamma is not None:
        # Case if output has two dimensions
        if output.data.ndim == 2:
            output = output * gamma.unsqueeze(dim=0)
        # Case if output has three dimensions
        else:
            output = output * gamma.unsqueeze(dim=0).unsqueeze(dim=-1)
    if beta is not None:
        # Case if output has two dimensions
        if output.data.ndim == 2:
            output = output + beta.unsqueeze(dim=0)
        # Case if output has three dimensions
        else:
            output = output + beta.unsqueeze(dim=0).unsqueeze(dim=-1)
    return output, running_mean, running_std


def batch_norm_2d(tensor: Tensor, gamma: Optional[Tensor] = None, beta: Optional[Tensor] = None,
                  mean: Optional[Tensor] = None, std: Optional[Tensor] = None,
                  eps: float = 1e-05, running_mean: Optional[Tensor] = None, running_std: Optional[Tensor] = None,
                  momentum: float = None) -> Tuple[Tensor, Optional[Tensor], Optional[Tensor]]:
    """
    Function implements a 2D batch normalization operation
    :param tensor: (Tensor) Input tensor
    :param gamma: (Optional[Tensor]) Learnable gamma factor which is multiplied to the output
    :param beta: (Optional[Tensor]) Learnable beta factor which is added to the output
    :param mean: (Optional[Tensor]) Mean for normalization
    :param std: (Optional[Tensor]) Variance for normalization
    :param eps: (float) Constant for numerical stability
    :param running_mean: (Optional[Tensor]) Running mean
    :param running_std: (Optional[Tensor]) Running std
    :param momentum: (Optional[float]) Momentum of running statistics
    :return: (Tuple[Tensor, Optional[Tensor], Optional[Tensor]]) Output tensor and optional running mean and std tensor
    """
    # Compute mean and std if not given
    if mean is None:
        mean = tensor.mean()
    if std is None:
        std = tensor.std()
    # If running statistics are given perform update
    if running_mean is not None and running_std is not None and momentum is not None:
        running_mean.data = momentum * running_mean.data + (1. - momentum) * mean.data
        running_std.data = momentum * running_std.data + (1. - momentum) * std.data
        mean.data = running_mean.data
        std.data = running_std.data
    # Apply normalization
    output = (tensor - mean) / (std + eps)
    # Apply learnable parameters if given
    if gamma is not None:
        output = output * gamma.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    if beta is not None:
        output = output + beta.unsqueeze(dim=0).unsqueeze(dim=-1).unsqueeze(dim=-1)
    return output, running_mean, running_std


def upsampling_nearest_2d(input: Tensor, scale_factor: int = 2) -> Tensor:
    """
    This function implements 2d nearest neighbour upsampling in autograd
    :param input: (Tensor) Input tensor of shape (batch size, channels, height, width)
    :param scale_factor: (int) Scaling factor
    :return: (Tensor) Output tensor of shape (batch size, channels, height * scale factor, width * scale factor)
    """
    # Check parameter
    assert scale_factor > 0, 'Scale factor must be greater than zero.'
    # Upsampling
    output = np.repeat(np.repeat(input.data, scale_factor, axis=-1), scale_factor, axis=-2)
    # Check grad
    requires_grad = input.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_upsampling_nearest_2d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Backward pass as max pooling
            height_factor = grad.shape[2] // scale_factor
            width_factor = grad.shape[2] // scale_factor
            # Perform max pooling
            grad_reshaped = grad.data[:, :, :height_factor * scale_factor, :width_factor * scale_factor] \
                .reshape(grad.shape[0], grad.shape[1], height_factor, scale_factor, width_factor, scale_factor)
            grad = grad_reshaped.max(axis=(3, 5))
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=input, grad_fn=grad_upsampling_nearest_2d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def upsampling_nearest_1d(input: Tensor, scale_factor: int = 2) -> Tensor:
    """
    This function implements nearest neighbour upsampling in autograd
    :param input: (Tensor) Input tensor of shape (batch size, channels, features)
    :param scale_factor: (int) Scaling factor
    :return: (Tensor) Output tensor of shape (batch size, channels, scale factor * features)
    """
    # Check parameter
    assert scale_factor > 0, 'Scale factor must be greater than zero.'
    # Upsampling
    output = np.repeat(input.data, scale_factor, axis=-1)
    # Check grad
    requires_grad = input.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_upsampling_nearest_1d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Backward pass as max pooling
            features_factor = grad.shape[-1] // scale_factor
            grad = grad[:, :, :features_factor * scale_factor] \
                .reshape(grad.shape[0], grad.shape[1], features_factor, scale_factor) \
                .max(axis=3)
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=input, grad_fn=grad_upsampling_nearest_1d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def _conv_1d_core(input: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Implements convolution operation in numpy
    :param input: (np.ndarray) Input array of shape (batch size, input channels, height, width)
    :param kernel: (np.ndarray) Kernel matrix of shape (output channels, input channels, kernel size, kernel size)
    :return: (np.ndarray) Output array
    """
    # Reshape input matrix
    sub_shape = (input.shape[2] - kernel.shape[2] + 1,)
    view_shape = input.shape[:2] + tuple(np.subtract(input.shape[2:], sub_shape) + 1) + sub_shape
    strides = input.strides[:2] + input.strides[2:] + input.strides[2:]
    sub_matrices = np.lib.stride_tricks.as_strided(input, view_shape, strides)
    # Perform convolution
    return np.einsum('oci, bcik->bok', kernel, sub_matrices)


def conv_1d(input: Tensor, kernel: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    This function implements a 1d convolution (cross-correlation) in autograd
    :param input: (Tensor) Input tensor of shape (batch size, input channels, input features)
    :param kernel: (Tensor) Kernel tensor of shape (output channels, input channels, kernel size)
    :param bias: (Tensor) Bias tensor of shape (output channels)
    :return: (Tensor) Output tensor of shape (batch size, output channels, output features)
    """
    # Check dimensions of parameters
    assert input.data.ndim == 3, 'Input tensor must have three dimensions.'
    assert kernel.data.ndim == 3, 'Kernel tensor must have three dimensions.'
    assert input.shape[1] == kernel.shape[1], 'Kernel features and input features must match.'
    # Perform convolution
    output = _conv_1d_core(input.data, kernel.data)
    # Check if gradient is required
    requires_grad = input.requires_grad or kernel.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if input.requires_grad:
        # Make gradient function
        def grad_conv1d_input(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            grad = _conv_1d_core(np.pad(grad, ((0, 0), (0, 0), (kernel.shape[2] - 1, kernel.shape[2] - 1)), 'constant',
                                        constant_values=(1)), kernel.data.transpose((1, 0, 2)))
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=input, grad_fn=grad_conv1d_input))

    if kernel.requires_grad:
        # Make gradient function
        def grad_conv1d_kernel(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            grad = _conv_1d_core(input.data.transpose((1, 0, 2)), grad.transpose((1, 0, 2))).transpose((1, 0, 2))
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=kernel, grad_fn=grad_conv1d_kernel))

    output_conv = Tensor(data=output, dependencies=dependencies, requires_grad=requires_grad)
    # Apply bias if utilized
    if bias is None:
        return output_conv
    assert bias.data.ndim == 1, 'Bias tensor must have three dimensions.'
    # Reshape bias tensor to match output of matrix multiplication
    bias_batched = np.expand_dims(np.expand_dims(bias.data, axis=0), axis=-1)
    # Perform addition
    output_bias_add = output_conv.data + bias_batched
    # Check if gradient is required
    requires_grad = output_conv.requires_grad or bias.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if output_conv.requires_grad:
        # Make gradient function
        def grad_conv_output(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """

            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_conv, grad_fn=grad_conv_output))

    if bias.requires_grad:
        # Make gradient function
        def grad_conv_bias(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_conv, grad_fn=grad_conv_bias))

    return Tensor(data=output_bias_add, dependencies=dependencies, requires_grad=requires_grad)


def linear(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    This function implements a linear layer in autograd
    :param input: (Tensor) Input tensor of shape (batch size, *, input features). * various number of channels
    :param weight: (Tensor) Weight tensor of shape (output features, input features)
    :param bias: (Tensor) Bias tensor of shape (output features)
    :return: (Tensor) Output tensor of shape (batch size, *, output features)
    """
    # Check dimensions of input tensors
    assert input.data.ndim in [2, 3], \
        'Input tensor has the wrong number of dimensions. Only two or three dimensions are supported'
    # Add third dimension to input tensor if needed
    if input.data.ndim == 2:
        # Add dimension in position one
        output = np.expand_dims(input.data, axis=1)
    else:
        output = input.data
    # Expend weights by batch size dimension
    weight_batched = np.expand_dims(weight.data, 0)
    # Perform matrix multiplication of weights and input
    output = weight_batched @ output.transpose((0, 2, 1))
    # Transpose output to get the desired output shape
    output = output.transpose((0, 2, 1))
    # Check if gradient is required
    requires_grad = input.requires_grad or weight.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if input.requires_grad:
        # Make gradient function
        def grad_linear_input(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad @ weight.data

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=input, grad_fn=grad_linear_input))

    if weight.requires_grad:
        # Make gradient function
        def grad_linear_weight(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            if input.data.ndim == 3:
                # Transpose gradient dimensions 1 and 2, perform mat mul and sum over batch size dimension
                return (grad.transpose((0, 2, 1)) @ input.data).sum(axis=0)
            return grad.T @ input.data

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=weight, grad_fn=grad_linear_weight))

    # Remove second dimension from output if needed
    if input.data.ndim == 2:
        output = output[:, 0, :]

    # Make output tensor of matrix multiplication
    output_mm = Tensor(data=output, dependencies=dependencies, requires_grad=requires_grad)

    # Perform bias addition
    if bias is None:
        return output_mm
    # Reshape bias tensor to match output of matrix multiplication
    if output_mm.data.ndim == 2:
        bias_batched = np.expand_dims(bias.data, axis=0)
    else:
        bias_batched = np.expand_dims(np.expand_dims(bias.data, axis=0), axis=0)
    # Perform addition
    output_bias_add = output_mm.data + bias_batched
    # Check if gradient is required
    requires_grad = output_mm.requires_grad or bias.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward function if needed
    if output_mm.requires_grad:
        # Make gradient function
        def grad_linear_output_mm(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_mm, grad_fn=grad_linear_output_mm))

    if bias.requires_grad:
        # Make gradient function
        def grad_linear_bias(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=output_mm, grad_fn=grad_linear_bias))

    return Tensor(data=output_bias_add, dependencies=dependencies, requires_grad=requires_grad)


def max_pool_1d(tensor: Tensor, kernel_size: int) -> Tensor:
    """
    This function implements a 1d max pooling operation in autograd
    :param tensor: (Tensor) Input tensor
    :param kernel_size: (Tuple[int]) Kernel size of the pooling operation.
    :return: (Tensor) Output tensor
    """
    # Check input dimensions
    assert tensor.data.ndim == 3, 'Input tensor must have three dimensions (batch size, channels, features).'
    # Get shape
    batch_size, channels, num_features = tensor.shape
    # Calc factors
    features_factor = num_features // kernel_size
    # Perform max pooling
    input_reshaped = tensor.data[:, :, :features_factor * kernel_size] \
        .reshape(batch_size, channels, features_factor, kernel_size)
    output = np.max(input_reshaped, axis=3)
    # Get indexes of max values
    indexes = (tensor.data == np.repeat(output, kernel_size, axis=2)).astype(float)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_max_pool_1d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Repeat gradient by kernel size
            unpooled_grad = np.repeat(grad.data, kernel_size, axis=2)
            # Mask out not used elements
            grad = unpooled_grad * indexes
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_max_pool_1d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def avg_pool_1d(tensor: Tensor, kernel_size: int) -> Tensor:
    """
    This function implements a 1d average pooling operation in autograd
    :param tensor: (Tensor) Input tensor
    :param kernel_size: (Tuple[int]) Kernel size of the pooling operation.
    :return: (Tensor) Output tensor
    """
    # Check input dimensions
    assert tensor.data.ndim == 3, 'Input tensor must have three dimensions (batch size, channels, features).'
    # Get shape
    batch_size, channels, num_features = tensor.shape
    # Calc factors
    features_factor = num_features // kernel_size
    # Perform max pooling
    input_reshaped = tensor.data[:, :, :features_factor * kernel_size] \
        .reshape(batch_size, channels, features_factor, kernel_size)
    output = np.mean(input_reshaped, axis=3, keepdims=False)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_max_pool_1d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            # Repeat gradient by kernel size
            unpooled_grad = np.repeat(grad.data, kernel_size, axis=2)
            # Mask out not used elements
            grad = (1 / kernel_size) * unpooled_grad
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_max_pool_1d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def dropout(tensor: Tensor, p: float = 0.2) -> Tensor:
    """
    Method performs dropout with a autograd tensor.
    :param tensor: (Tensor) Input tensor
    :param p: (float) Probability that a activation element is set to zero
    :return: (Tensor) Output tensor
    """
    # Check argument
    assert 0.0 <= p <= 1.0, 'Parameter p must be in the range of [0, 1].'
    # Apply dropout
    mask = (np.random.randint(0, 1, size=tensor.shape) > p).astype(float)
    output = tensor.data * mask
    # Check if grad is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: grad * mask)] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def dropout2d(tensor: Tensor, p: float = 0.2) -> Tensor:
    """
    Method performs 2D channel-wise dropout with a autograd tensor.
    :param tensor: (Tensor) Input tensor
    :param p: (float) Probability that a activation element is set to zero
    :return: (Tensor) Output tensor
    """
    # Check argument
    assert 0.0 <= p <= 1.0, 'Parameter p must be in the range of [0, 1].'
    # Apply dropout
    mask = (np.random.randint(0, 2, size=tensor.shape[0]) > p).astype(float).reshape(1, -1, 1, 1)
    output = tensor.data * mask
    # Check if grad is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: grad * mask)] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def softmax(tensor: Tensor, axis: int = 1) -> Tensor:
    """
    This function implements the softmax activation in autograd
    :param tensor: (Tensor) Input tensor
    :param axis: (int) Axis to apply softmax
    :return: (Tensor) Output tensor
    """
    output_exp = autograd.exp(tensor)
    output = output_exp / (autograd.sum(output_exp, axis=axis, keepdims=True))
    return output


def cross_entropy_loss(prediction: Tensor, label: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Function implements the multi class cross entropy loss in autograd
    :param prediction: (Tensor) Prediction tensor
    :param label: (Tensor) One hot encoded label tensor
    :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
    :return: (Tensor) Loss value
    """
    # Assert shapes to be the same for prediction and label
    assert label.shape == prediction.shape, 'Shape of label must match with prediction'
    # Compute loss
    loss = - (label * autograd.log(prediction))
    # Apply reduction
    return _apply_reduction(tensor=loss, reduction=reduction)


def l1_loss(prediction: Tensor, label: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Function implements the l1 loss in autograd
    :param prediction: (Tensor) Prediction tensor
    :param label: (Tensor) One hot encoded label tensor
    :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
    :return: (Tensor) Loss value
    """
    # Assert shapes to be the same for prediction and label
    assert label.shape == prediction.shape, 'Shape of label must match with prediction'
    # Compute loss
    loss = autograd.abs(prediction - label)
    # Apply reduction
    return _apply_reduction(tensor=loss, reduction=reduction)


def mse_loss(prediction: Tensor, label: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Function implements the mean squared error loss in autograd
    :param prediction: (Tensor) Prediction tensor
    :param label: (Tensor) One hot encoded label tensor
    :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
    :return: (Tensor) Loss value
    """
    # Assert shapes to be the same for prediction and label
    assert label.shape == prediction.shape, 'Shape of label must match with prediction'
    # Compute loss
    loss = autograd.abs(prediction - label) ** 2
    # Apply reduction
    return _apply_reduction(tensor=loss, reduction=reduction)


def binary_cross_entropy_loss(prediction: Tensor, label: Tensor, reduction: str = 'mean') -> Tensor:
    """
    This function implements the binary cross entropy loss in autograd
    :param prediction: (Tensor) Prediction tensor
    :param label: (Tensor) One hot encoded label tensor
    :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
    :return: (Tensor) Loss value
    """
    # Check shape of prediction and label
    assert label.shape == prediction.shape, 'Shape of label must match with prediction'
    # Compute loss
    loss = - (label * autograd.log(prediction) + (1 - label) * autograd.log(1 - prediction))
    # Apply reduction
    return _apply_reduction(tensor=loss, reduction=reduction)


def softmax_cross_entropy_loss(prediction: Tensor, label: Tensor, reduction: str = 'mean', axis: int = 1) -> Tensor:
    """
    Function implements the softmax multi class cross entropy loss in autograd
    :param prediction: (Tensor) Prediction tensor
    :param label: (Tensor) One hot encoded label tensor
    :param reduction: (str) Type of reduction to perform after apply the loss (mean, sum or none)
    :param axis: (int) Axis to apply softmax
    :return: (Tensor) Loss value
    """
    # Assert tha label dose not require gradient
    assert not label.requires_grad, 'Gradient for label not supported. Use normal cross entropy loss, instead.'
    # Apply softmax
    prediction_exp = np.exp(prediction.data - prediction.data.max())
    prediction_softmax = prediction_exp / (np.sum(prediction_exp, axis=axis, keepdims=True) + 1e-18)
    # Compute loss
    loss = - (label.data * np.log(np.clip(prediction_softmax, 1e-18, 1.0 - 1e-18)))
    # Check if gradient is needed
    requires_grad = prediction.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_scel(grad: np.ndarray) -> np.ndarray:
            """
            Function computes gradient of the sin function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            """
            return grad * (prediction_softmax - label.data)

        dependency = [Dependency(activation=prediction, grad_fn=grad_scel)]
    else:
        dependency = None
    # Build loss tensor
    loss = Tensor(data=loss, dependencies=dependency, requires_grad=requires_grad)
    # Apply reduction
    return _apply_reduction(tensor=loss, reduction=reduction)


def _apply_reduction(tensor: Tensor, reduction: str) -> Tensor:
    """
    Function apply a given reduction (mean, sum or none) to a given tensor
    :param tensor: (Tensor) Input tensor
    :param reduction: (str) Type of reduction
    :return: (Tensor) Output tensor
    """
    # Check parameter
    assert reduction in ['mean', 'sum', 'none'], 'Reduction {} is not available. Use `mean`, `sum` or `none`'
    # Apply reduction
    if reduction == 'mean':
        return tensor.mean()
    elif reduction == 'sum':
        return tensor.sum()
    else:
        return tensor


def pau(tensor: Tensor, m: Tensor, n: Tensor) -> Tensor:
    """
    Implementation of the pade activation unit in autograd
    :param tensor: (Tensor) Input tensor of any shape
    :param m: (Tensor) Parameters for the numerator polynomial
    :param n: (Tensor) Parameters for the denominator polynomial
    :return: (Tensor) Activated output tensor
    """
    # Forward pass
    output_numerator = m.data[0] \
                       + tensor.data * m.data[1] \
                       + (tensor.data ** 2 * m.data[2]) \
                       + (tensor.data ** 3 * m.data[3]) \
                       + (tensor.data ** 4 * m.data[4]) \
                       + (tensor.data ** 5 * m.data[5])
    output_denominator_a = tensor.data * n.data[0] \
                           + (tensor.data ** 2 * n.data[1]) \
                           + (tensor.data ** 3 * n.data[2]) \
                           + (tensor.data ** 4 * n.data[3])
    output_denominator = 1. + np.abs(output_denominator_a)
    output = output_numerator / output_denominator
    # Check if gradient is required
    requires_grad = tensor.requires_grad or m.requires_grad or n.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward operations to output tensor
    if tensor.requires_grad:
        # Make gradient function for gradient with respect to input tensor
        def grad_pade_input(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Calc gradient dP(x) / dx
            grad_p_x = m[1] \
                       + (2. * tensor.data * m.data[2]) \
                       + (3. * tensor.data ** 2 * m.data[3]) \
                       + (4. * tensor.data ** 3 * m.data[4]) \
                       + (5. * tensor.data ** 4 * m.data[5])
            # Calc gradient dQ(x) / dx
            grad_q_x = np.sign(output_denominator_a) * n.data[0] \
                       + (2. * tensor.data * n.data[1]) \
                       + (3. * tensor.data ** 2 * n.data[2]) \
                       + (4. * tensor.data ** 3 * n.data[3]) \
                # Combine parts and input grad
            return (grad_p_x * (1. / output_denominator)
                    - grad_q_x * (output_numerator / output_denominator ** 2)) * grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=tensor, grad_fn=grad_pade_input))

    if m.requires_grad:
        # Make gradient function for gradient with respect to m
        def grad_pade_m(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Compute grad of m
            grad_m = np.empty(m.shape)
            grad_m[0] = (1 / output_denominator).sum()
            grad_m[1] = (tensor.data / output_denominator).sum()
            grad_m[2] = (tensor.data ** 2 / output_denominator).sum()
            grad_m[3] = (tensor.data ** 3 / output_denominator).sum()
            grad_m[4] = (tensor.data ** 4 / output_denominator).sum()
            grad_m[5] = (tensor.data ** 5 / output_denominator).sum()
            # Apply chain rule
            return grad_m * grad.sum()

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=m, grad_fn=grad_pade_m))

    if n.requires_grad:
        # Make gradient function for gradient with respect to m
        def grad_pade_n(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Compute grad of n
            grad_n = np.empty(n.shape)
            grad_n[0] = (- tensor.data * np.sign(output_denominator_a)
                         * (output_numerator / output_denominator ** 2)).sum()
            grad_n[1] = (- tensor.data ** 2 * np.sign(output_denominator_a)
                         * (output_numerator / output_denominator ** 2)).sum()
            grad_n[2] = (- tensor.data ** 3 * np.sign(output_denominator_a)
                         * (output_numerator / output_denominator ** 2)).sum()
            grad_n[3] = (- tensor.data ** 4 * np.sign(output_denominator_a)
                         * (output_numerator / output_denominator ** 2)).sum()
            # Apply chain rule
            return grad_n * grad.sum()

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=n, grad_fn=grad_pade_n))
    # Make output autograd tensor
    output = Tensor(data=output, dependencies=dependencies, requires_grad=requires_grad)
    return output
