import numpy as np

from autograd.tensor import Tensor, Dependency


def exp(tensor: Tensor) -> Tensor:
    '''
    Function implements the element-wise exponential function
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply exp
    output = np.exp(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_exp(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the sin function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * np.exp(tensor.data)

        dependency = [Dependency(activation=tensor, grad_fn=grad_exp)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def log(tensor: Tensor) -> Tensor:
    '''
    Function implements the natural logarithm in autograd.
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply exp
    output = np.log(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_log(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the sin function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (1 / (tensor.data))

        dependency = [Dependency(activation=tensor, grad_fn=grad_log)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def sqrt(tensor: Tensor) -> Tensor:
    '''
    Function implements the square root in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    return tensor ** 0.5


def sin(tensor: Tensor) -> Tensor:
    '''
    Function implements the sin function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply sin
    output = np.sin(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_sin(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the sin function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * np.cos(tensor.data)

        dependency = [Dependency(activation=tensor, grad_fn=grad_sin)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def cos(tensor: Tensor) -> Tensor:
    '''
    Function implements the cos function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply cos
    output = np.cos(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_cos(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the cos function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (- np.sin(tensor.data))

        dependency = [Dependency(activation=tensor, grad_fn=grad_cos)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def tan(tensor: Tensor) -> Tensor:
    '''
    Function implements the tan function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply cos
    output = np.tan(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_tan(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the cos function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (1 / (np.cos(tensor.data) ** 2))

        dependency = [Dependency(activation=tensor, grad_fn=grad_tan)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def tanh(tensor: Tensor) -> Tensor:
    '''
    Function implements the tanh function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply tanh
    output = np.tanh(tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_tanh(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the tanh function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (1 - output * output)

        dependency = [Dependency(activation=tensor, grad_fn=grad_tanh)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def softplus(tensor: Tensor) -> Tensor:
    '''
    Function implements the softplus function in autograd.
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply softplus
    output = np.log(1.0 + np.exp(tensor.data))
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_softplus(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes the gradient of the elu function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (1 / (1 + np.exp(- tensor.data)))

        dependency = [Dependency(activation=tensor, grad_fn=grad_softplus)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def elu(tensor: Tensor, alpha: float = 1.0) -> Tensor:
    '''
    Function implements the elu function in autograd
    :param tensor: (Tensor) Input tensor
    :param alpha: (float) Alpha parameter of exponential slope
    :return: (Tensor) Output Tensor
    '''
    # Apply elu
    output = np.where(tensor.data > 0.0, tensor.data, alpha * (np.exp(tensor.data) - 1))
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_elu(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes the gradient of the elu function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (np.where(tensor.data > 0.0, 1.0, alpha * np.exp(tensor.data)))

        dependency = [Dependency(activation=tensor, grad_fn=grad_elu)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def leaky_relu(tensor: Tensor, negative_slope: float = 0.2) -> Tensor:
    '''
    Function implements the leaky-relu function in autograd
    :param tensor: (Tensor) Input tensor
    :param negative_slope: (float) Negative slope of leaky-relu
    :return: (Tensor) Output Tensor
    '''
    # Apply leaky-relu
    output = np.maximum(tensor.data, negative_slope * tensor.data)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_relu(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes the gradient of the leaky-relu function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * np.where(output > 0.0, 1.0, negative_slope)

        dependency = [Dependency(activation=tensor, grad_fn=grad_relu)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)


def relu(tensor: Tensor) -> Tensor:
    '''
    Function implements the relu function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output Tensor
    '''
    return leaky_relu(tensor=tensor, negative_slope=0.0)


def selu(tensor: Tensor) -> Tensor:
    '''
    Function implements the selu function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output Tensor
    '''
    # Apply elu
    output = elu(tensor=tensor, alpha=1.6732632423543772848170429916717)
    # Apply scale
    output = 1.0507009873554804934193349852946 * output
    return output


def sigmoid(tensor: Tensor) -> Tensor:
    '''
    Function implements the sigmoid function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    '''
    # Apply sigmoid
    output = 1 / (1 + np.exp(- tensor.data))
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Make backward function if needed
    if requires_grad:
        def grad_sigmoid(grad: np.ndarray) -> np.ndarray:
            '''
            Function computes gradient of the sigmoid function
            :param grad: (Tensor) Previous gradient
            :return: (Tensor) Gradient
            '''
            return grad * (output * (1 - output))

        dependency = [Dependency(activation=tensor, grad_fn=grad_sigmoid)]
    else:
        dependency = None

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependency)
