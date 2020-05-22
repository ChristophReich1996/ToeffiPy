from __future__ import annotations
from typing import List, Union, Optional, Callable, Tuple, Dict
from dataclasses import dataclass

import numpy as np


@dataclass
class Dependency:
    """
    Data class including activation and gradient function
    """
    activation: Tensor
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor(object):
    """
    This class implements an autograd tensor
    """

    def __init__(self,
                 data: Union[int, list, float, np.ndarray, Tensor],
                 requires_grad: bool = False,
                 dependencies: List = None) -> None:
        """
        Constructor method
        :param data: (int, list, float, np.ndarray, Tensor) Data of the tensor
        :param requires_grad: (bool)
        :param dependencies: (List[Dependency])
        """
        # Save parameter
        self._data = Tensor.__make_ndarray(data)
        self.requires_grad = requires_grad
        self.dependencies = dependencies or []
        self.shape = self._data.shape
        # Init gradient
        self.grad: Optional[Tensor] = None
        if self.requires_grad:
            self.zero_grad()

    def zero_grad(self) -> None:
        """
        Method sets gradients to zero
        """
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data, dtype=float))
        else:
            self.grad.data.fill(0.0)

    @property
    def data(self) -> np.ndarray:
        """
        Getter method for _data
        :return: (np.ndarray) Data of tensor
        """
        return self._data

    @data.setter
    def data(self, data: Union[int, list, float, np.ndarray, Tensor]) -> None:
        """
        Method sets data field of tensor
        :param data: (int, list, float, np.ndarray, Tensor) Data of the tensor
        """
        self._data = self.__make_ndarray(data)
        # Reset gradient since data is updated
        self.zero_grad()
        # Set size filed
        self.shape = self._data.shape

    @staticmethod
    def __make_ndarray(data: Union[int, list, float, np.ndarray, Tensor]) -> np.ndarray:
        """
        Method converts a given data field to a np.ndarray, including floats.
        :param data: (int, list, float, np.ndarray, Tensor) Data field
        :return: (np.ndarray) Numpy array including given data
        """
        # Case of, data is a np.ndarray
        if isinstance(data, np.ndarray):
            # Ensure np.ndarray includes floats
            if data.dtype != float:
                # Cast data
                data.astype(float)
            return data
        # Case of, data is a tensor
        elif isinstance(data, Tensor):
            return data.data
        # Case of, data is a list, float or int
        else:
            return np.array(data, dtype=float)

    @staticmethod
    def __ensure_tensor(data_object: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method converts a given data object to a tensor
        :param data_object: (Tensor, int, list, float, np.ndarray) Data object
        :return: (Tensor) Tensor object
        """
        if isinstance(data_object, Tensor):
            return data_object
        else:
            return Tensor(data_object)

    def __repr__(self) -> str:
        """
        Method returns information of the tensor
        :return: (str) Representation of the tensor as a string
        """
        return 'Tensor({}, requires_grad={}, shape={})'.format(self.data, self.requires_grad, self.shape)

    def backward(self, grad: Tensor = None) -> None:
        """
        Method computes the gradient of the tensor
        :param grad: (Tensor) Previous gradient
        """
        assert self.requires_grad, 'Backward was called on a non-required-grad tensor'
        # Check grad
        if grad is None:
            if self.shape == ():
                # Set grad
                grad = Tensor(1.0)
            else:
                raise RuntimeError('Grad must be specified for a non-scalar-tensor.')
        # Add gradients if needed
        self.grad.data = self.grad.data + grad.data
        # Perform backprop
        for dependency in self.dependencies:
            backward_grad = dependency.grad_fn(grad.data)
            dependency.activation.backward(Tensor(backward_grad))

    def __add__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable addition. self + other.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return add(self, Tensor.__ensure_tensor(other))

    def __radd__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable addition. other + self.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return add(Tensor.__ensure_tensor(other), self)

    def __iadd__(self, other: Union[int, list, float, np.ndarray, Tensor]) -> Tensor:
        """
        Method performs inplace addition. self += other. Gradients not supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        # Perform in place addition which invalids the gradient
        self.data = self.data + Tensor.__make_ndarray(other)
        return self

    def __sub__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable subtraction. self + other.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return sub(self, Tensor.__ensure_tensor(other))

    def __rsub__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable subtraction. other + self.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return sub(Tensor.__ensure_tensor(other), self)

    def __isub__(self, other: Union[int, list, float, np.ndarray, Tensor]) -> Tensor:
        """
        Method performs inplace subtraction. self -= other. Gradients not supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        # Perform in place addition which invalids the gradient
        self.data = self.data - Tensor.__make_ndarray(other)
        return self

    def __mul__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable multiplication. self * other.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return mul(self, Tensor.__ensure_tensor(other))

    def __rmul__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable multiplication. other * self.
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return mul(Tensor.__ensure_tensor(other), self)

    def __imul__(self, other: Union[int, list, float, np.ndarray, Tensor]) -> Tensor:
        """
        Method performs inplace multiplication. self *= other. Gradients not supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        # Perform in place addition which invalids the gradient
        self.data = self.data * Tensor.__make_ndarray(other)
        return self

    def __truediv__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method perform differentiable element-wise division. self / other. Batching supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return div(self, Tensor.__ensure_tensor(other))

    def __rtruediv__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method perform differentiable element-wise division. other / self. Batching supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return div(Tensor.__ensure_tensor(other), self)

    def __itruediv__(self, other: Union[int, list, float, np.ndarray, Tensor]) -> Tensor:
        """
        Method performs inplace division. self /= other. Gradients not supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        # Perform in place addition which invalids the gradient
        self.data = self.data / Tensor.__make_ndarray(other)
        return self

    def __matmul__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable matrix multiplication. self @ other. Batching supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return matmul(self, Tensor.__ensure_tensor(other))

    def __rmatmul__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Method performs differentiable matrix multiplication. self @ other. Batching supported!
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Result
        """
        return matmul(Tensor.__ensure_tensor(other), self)

    def __neg__(self) -> Tensor:
        """
        Method performs negation of a autograd tensor.
        :return: (Tensor) Inverted output tensor
        """
        return neg(self)

    def __pow__(self, power: Union[int, float]) -> Tensor:
        """
        Method applies an element-wise power to a tensor.
        :param power: (int, float) Exponent
        :return: (Tensor) Output tensor
        """
        return pow(self, power)

    def __abs__(self) -> Tensor:
        """
        Method apply the absolute function to a tensor.
        :return: (Tensor) Output tensor
        """
        return abs(self)

    def __getitem__(self, indexes: Union[Tuple[slice, ...], slice, Tuple[int, ...], int]) -> Tensor:
        """
        Get item method.
        :param indexes: (Any) Indexes
        :return: (Tensor) Output tensor
        """
        return _slice(self, indexes)

    def __eq__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the equal comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data == Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __ne__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the not equal comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data != Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __lt__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the less-than comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data < Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __gt__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the greater-than comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data > Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __le__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the less-than-or-equal-to comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data <= Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __ge__(self, other: Union[Tensor, int, list, float, np.ndarray]) -> Tensor:
        """
        Implements the greater-than-or-equal-to comparison function
        :param other: (Tensor, int, list, float, np.ndarray) Other tensor like object
        :return: (Tensor) Output tensor
        """
        self.data = (self.data >= Tensor.__ensure_tensor(other).data).astype(float)
        return self

    def __round__(self, n: int = 0) -> Tensor:
        """
        Method implements the round function
        :param n: (int) Decimals to round
        :return: (Tensor) Output tensor
        """
        self.data = np.round(self.data, decimals=n)
        return self

    def sum(self, axis: int = None, keepdims: bool = False) -> Tensor:
        """
        Method sums up a given tensor.
        :param axis: (int) Axis to apply summation
        :param keepdims: (bool) If true summed up dimensions are retained
        :return: (Tensor) Output tensor
        """
        return sum(self, axis=axis, keepdims=keepdims)

    def mean(self) -> Tensor:
        """
        Method computes the mean of a tensor.
        :return: (Tensor) Output tensor
        """
        return mean(self)

    def var(self) -> Tensor:
        """
        Method computes the variance of a tensor.
        :return: (Tensor) Output tensor
        """
        return var(self)

    def std(self) -> Tensor:
        """
        Method computes the standard deviation of a tensor.
        :return: (Tensor) Output tensor
        """
        return std(self)

    def clone(self) -> Tensor:
        """
        Method clones a given tensor but input and output tensors remain in graph
        :return: (Tensor) Output tensor
        """
        return clone(self)

    def unsqueeze(self, dim: int = -1) -> Tensor:
        """
        Method adds a dimension to the tensor into the given position.
        :param dim: (int) Position to add the dimension
        :return: (Tensor) Output tensor
        """
        return unsqueeze(self, dim=dim)

    def squeeze(self, dim: int = -1) -> Tensor:
        """
        Method removes a dimension to the tensor at a given position.
        :param dim: (int) Position to add the dimension
        :return: (Tensor) Output tensor
        """
        return squeeze(self, dim=dim)

    def max(self) -> float:
        """
        Method returns the max value of the tensor
        :return: (float) Tensor
        """
        return max(self)


def max(tensor: Tensor) -> float:
    """
    Implementation of the max function for autograd tensors
    :param tensor: (Tensor) Input tensor
    :return: (float) Max value
    """
    return float(tensor.data.max())


def abs(tensor: Tensor) -> Tensor:
    """
    Function implements the absolute function in autograd.
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    """
    # Perform abs
    output = np.abs(tensor.data)
    # Check if grad is needed
    requires_grad = tensor.requires_grad
    # Add backward functions if needed
    if tensor.requires_grad:
        # Make gradient function
        def grad_abs(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            grad = grad * np.sign(tensor.data)
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_abs)]

    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def pow(tensor: Tensor, power: Union[int, float]) -> Tensor:
    """
    Function implements the element-wise power function in autograd.
    :param tensor: (Tensor) Input tensor
    :param power: (int, float) Exponent
    :return: (Tensor) Output tensor
    """
    # Perform power
    output = tensor.data ** power
    # Check if grad is needed
    requires_grad = tensor.requires_grad
    # Add backward functions if needed
    if tensor.requires_grad:
        # Make gradient function
        def grad_pow(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            grad = grad * (power * (tensor.data) ** (power - 1))
            return grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_pow)]

    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def div(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Function implements a differentiable elemnt wise division of two tensors. Broadcasting is supported.
    :param tensor_1: (Tensor) First tensor
    :param tensor_2: (Tensor) Second tensor
    :return: (Tensor) Output tensor
    """
    # Invert second tensor 1 / tensor_2
    tensor_2_inverted = pow(tensor_2, -1)
    # Multiply tensor 1 with inverted tensor
    output = tensor_1 * tensor_2_inverted
    return output


def mul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Function implements a differentiable multiplication of two tensors. Broadcasting is supported.
    :param tensor_1: (Tensor) First tensor
    :param tensor_2: (Tensor) Second tensor
    :return: (Tensor) Output tensor
    """
    # Perform multiplication
    output = tensor_1.data * tensor_2.data
    # Check if grad is needed
    requires_grad = tensor_1.requires_grad or tensor_2.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward functions if needed
    if tensor_1.requires_grad:
        # Make gradient function
        def grad_mul_1(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            grad = grad * tensor_2.data
            # Get added dimensions
            dimensions = grad.ndim - tensor_1.data.ndim
            # Sum out added dimensions
            for _ in range(dimensions):
                grad = grad.sum(axis=0)
            # Sum across broadcasted dimensions
            for index, dimension in enumerate(tensor_1.shape):
                if dimension == 1:
                    grad = grad.sum(axis=index, keepdims=True)
            return grad

        # Make dependencies
        dependencies.append(Dependency(activation=tensor_1, grad_fn=grad_mul_1))

    if tensor_2.requires_grad:
        # Make gradient function
        def grad_mul_2(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            grad = grad * tensor_1.data
            # Get added dimensions
            dimensions = grad.ndim - tensor_2.data.ndim
            # Sum out added dimensions
            for _ in range(dimensions):
                grad = grad.sum(axis=0)
            # Sum across broadcasted dimensions
            for index, dimension in enumerate(tensor_2.shape):
                if dimension == 1:
                    grad = grad.sum(axis=index, keepdims=True)
            return grad

        # Make dependencies
        dependencies.append(Dependency(activation=tensor_2, grad_fn=grad_mul_2))

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def sum(tensor: Tensor, axis: int = None, keepdims: bool = False) -> Tensor:
    """
    Sums up a tensor to a scalar value
    :param tensor: (Tensor) Input tensor
    :param axis: (int) Axis to apply summation
    :param keepdims: (bool) If true summed up dimensions are retained
    :return: (Tensor) Output tensor (scalar)
    """
    # Save originale shape
    original_shape = tensor.shape
    # Perform summation
    output = tensor.data.sum(axis=axis, keepdims=keepdims)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_sum(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad * np.ones(original_shape)

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_sum)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def mean(tensor: Tensor) -> Tensor:
    """
    Function implements the mean function in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output scalar tensor
    """
    # Get number of elements in tensor
    number_of_tensor_elements = tensor.data.size
    # Sum up tensor
    output = tensor.sum()
    # Apply factor
    output = (1 / number_of_tensor_elements) * output
    return output


def var(tensor: Tensor) -> Tensor:
    """
    Function implements the variance function in autograd.
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Scalar output tensor
    """
    # Calc mean
    tensor_mean = mean(tensor)
    # Calc var
    variance = mean((tensor - tensor_mean) ** 2)
    return variance


def std(tensor: Tensor) -> Tensor:
    """
    Function implements the standard deviation in autograd.
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Scalar output tensor
    """
    return var(tensor) ** 0.5


def add(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Function implements a differentiable addition of two tensors. Broadcasting is supported.
    :param tensor_1: (Tensor) First tensor
    :param tensor_2: (Tensor) Second tensor
    :return: (Tensor) Output tensor
    """
    # Perform element wise addition
    output = tensor_1.data + tensor_2.data
    # Check if gradient is required
    requires_grad = tensor_1.requires_grad or tensor_2.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward operations to output tensor
    if tensor_1.requires_grad:
        # Make gradient function
        def grad_add_1(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Get added dimensions
            dimensions = grad.ndim - tensor_1.data.ndim
            # Sum out added dimensions
            for _ in range(dimensions):
                grad = grad.sum(axis=0)
            # Sum across broadcasted dimensions
            for index, dimension in enumerate(tensor_1.shape):
                if dimension == 1:
                    grad = grad.sum(axis=index, keepdims=True)
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=tensor_1, grad_fn=grad_add_1))

    if tensor_2.requires_grad:
        # Make gradient function
        def grad_add_2(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Get added dimensions
            dimensions = grad.ndim - tensor_2.data.ndim
            # Sum out added dimensions
            for _ in range(dimensions):
                grad = grad.sum(axis=0)
            # Sum across broadcasted dimensions
            for index, dimension in enumerate(tensor_2.shape):
                if dimension == 1:
                    grad = grad.sum(axis=index, keepdims=True)
            return grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=tensor_2, grad_fn=grad_add_2))

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def neg(tensor: Tensor) -> Tensor:
    """
    Function negates a given tensor in autograd
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    """
    # Calc output
    output = - tensor.data
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: -grad)] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def sub(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Function implements a differentiable subtraction
    :param tensor_1: (Tensor) First tensor
    :param tensor_2: (Tensor) Second tensor
    :return: (Tensor) Output tensor
    """
    return tensor_1 + (- tensor_2)


def matmul(tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """
    Function implements a differentiable matrix multiplication of two tensors.
    :param tensor_1: (Tensor) First Tensor
    :param tensor_2: (Tensor) Second Tensor
    :return: (Tensor) Result tensor
    """
    # Perform matrix multiplication
    output = tensor_1.data @ tensor_2.data
    # Check if gradient is required
    requires_grad = tensor_1.requires_grad or tensor_2.requires_grad
    # Init dependencies
    dependencies: List[Dependency] = []
    # Add backward operations to output tensor
    if tensor_1.requires_grad:
        # Make gradient function for gradient with respect to tensor 1
        def grad_matmul_1(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            # Deal with batched tensor
            if tensor_2.data.ndim > 2:
                return grad @ tensor_2.data.transpose(0, 2, 1)
            # Deal with non-batched tensor
            else:
                return grad @ tensor_2.data.T

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=tensor_1, grad_fn=grad_matmul_1))

    if tensor_2.requires_grad:
        # Make gradient function for gradient with respect to tensor 2
        def grad_matmul_2(grad: np.ndarray) -> np.ndarray:
            """
            Gradient function
            :param grad: (np.ndarray) Gradient with respect to the output
            :return: (np.ndarray) Gradient
            """
            if tensor_1.data.ndim > 2:
                return tensor_1.data.transpose(0, 2, 1) @ grad
            else:
                return tensor_1.data.T @ grad

        # Add grad function to dependencies
        dependencies.append(Dependency(activation=tensor_2, grad_fn=grad_matmul_2))

    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def _slice(tensor: Tensor, indexes: Union[Tuple[slice, ...], slice]) -> Tensor:
    """
    Function implements slicing of a autograd tensor
    :param tensor: (Tensor) Input tensor
    :param indexes: (Tuple[slice, ...], slice) Slice or Slices for indexing
    :return: (Tensor) Output tensor
    """
    # Apply indexes
    output = tensor.data[indexes]
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_slice(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            new_grad = np.zeros_like(output)
            new_grad[indexes] = grad
            return new_grad

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_slice)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def unsqueeze(tensor: Tensor, dim: int = -1) -> Tensor:
    """
    Function adds a dimension to a given tensor
    :param tensor: (Tensor) Input tensor
    :param dim: (int) Position where the new dimension is places
    :return: (Tensor) Output tensor with increased dimensionality of one
    """
    # Add dim
    output = np.expand_dims(tensor.data, axis=dim)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: grad.squeeze(axis=dim))] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def squeeze(tensor: Tensor, dim: int = -1) -> Tensor:
    """
    Function removes a dimension of a given tensor
    :param tensor: (Tensor) Input tensor
    :param dim: (int) Position where the new dimension is places
    :return: (Tensor) Output tensor with increased dimensionality of one
    """
    # Add dim
    output = np.squeeze(tensor.data, axis=dim)
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: np.expand_dims(grad, axis=dim))] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def clone(tensor: Tensor) -> Tensor:
    """
    Function makes a copy of a given tensor but input tensor remains in graph
    :param tensor: (Tensor) Input tensor
    :return: (Tensor) Output tensor
    """
    # Calc output
    output = tensor.data
    # Check if gradient is needed
    requires_grad = tensor.requires_grad
    # Add grad function
    dependencies = [Dependency(tensor, lambda grad: grad)] if requires_grad else None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def stack(tensors: List[Tensor], dim: int = 0) -> Tensor:
    """
    Function stacks a list of tensor up to one tensor. Autograd is not supported!
    :param tensors: (List[Tensor]) List of tensors
    :param dim: (int) Dimension to stack
    :return: (Tensor) Output tensor
    """
    # Init list of ndarrays
    ndarrays = []
    for tensor in tensors:
        if tensor.requires_grad:
            raise Warning('Stack function was called on a required grad tensor!')
        ndarrays.append(tensor.data)
    # Stack tensors and make new autograd tensor
    return Tensor(data=np.stack(ndarrays, axis=dim))


def pad_1d(tensor: Tensor, pad_width: Tuple[int, int], value: float = 0.0) -> Tensor:
    """
    Function applies padding to a given 1d tensor (batch size, *, features), * various dimension
    :param tensor: (Tensor) Input tensor of shape (batch size, *, features)
    :param pad_width: (Tuple[int, int]) Number of values padded to the edges of each axis
    :param value: (float) Padding value
    :return: (Tensor) Output tensor (batch size, *, features + padded width)
    """
    # Apply padding
    if tensor.data.ndim == 2:
        output = np.pad(tensor.data, pad_width=((0, 0), pad_width), mode='constant', constant_values=value)
    else:
        output = np.pad(tensor.data, pad_width=((0, 0), (0, 0), pad_width), mode='constant', constant_values=value)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_pad_1d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            if grad.ndim == 2:
                return grad[:, pad_width[0]:-pad_width[1]]
            return grad[:, :, pad_width[0]:-pad_width[1]]

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_pad_1d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def pad_2d(tensor: Tensor, pad_width: Tuple[Tuple[int, int], Tuple[int, int]], value: float = 0.0) -> Tensor:
    """
    Function applies padding to a given 2d tensor (batch size, channels, height, width)
    :param tensor: (Tensor) Input tensor of shape (batch size, channels, height, width)
    :param pad_width: (Tuple[Tuple[int, int], Tuple[int, int]]) Number of values padded to the edges of each axis
    :param value: (float) Padding value
    :return: (Tensor) Output tensor (batch size, channels, height + padded width, width + padded width)
    """
    # Apply padding
    output = np.pad(tensor.data, pad_width=((0, 0), (0, 0), pad_width[0], pad_width[1]), mode='constant',
                    constant_values=value)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_pad_2d(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad[:, :, pad_width[0][0]:-pad_width[0][1], pad_width[1][0]:-pad_width[1][1]]

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_pad_2d)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def flatten(tensor: Tensor, starting_dim: int = 1) -> Tensor:
    """
    This function implements a flatten operation which reshapes (flattens) a given tensor
    :param tensor: (Tensor) Input tensor
    :param starting_dim: (int) Dimension to start flattening
    :return: (Tensor) Flattened output tensor
    """
    # Save original shape
    original_shape = tensor.shape
    target_shape = original_shape[:starting_dim] + (-1,)
    # Flatten tensor
    output = tensor.data.reshape(target_shape)
    # Check grad
    requires_grad = tensor.requires_grad
    # Add backward function if needed
    if requires_grad:
        # Make gradient function
        def grad_flatten(grad: np.ndarray) -> np.ndarray:
            """
            Computes the gradient
            :param grad: (np.ndarray) Original gradient
            :return: (np.ndarray) Final gradient
            """
            return grad.reshape(original_shape)

        # Make dependencies
        dependencies = [Dependency(activation=tensor, grad_fn=grad_flatten)]
    else:
        dependencies = None
    return Tensor(data=output, requires_grad=requires_grad, dependencies=dependencies)


def save(tensor: Union[Tensor, Dict[str, Tensor]], path: str) -> None:
    """
    Function save an autograd tensor or a dict of autograd tensors
    :param tensor: (Tensor, Dict[str, Tensor]) Tensor or dict of tensors
    :param path: (str) Path and file name to store object
    """
    # Case if a simple tensor is given
    if isinstance(tensor, Tensor):
        np.save(path, tensor.data)
    # Case if dict of tensors is given
    else:
        np.savez(path, **tensor)


def load(path: str) -> Union[Tensor, np.lib.npyio.NpzFile]:
    """
    Function loads single tensor or a dict of tensors previously saved as .np or .npz.
    :param path: (str) Path to file to be loaded
    :return: (Tensor, np.lib.npyio.NpzFile) Loaded tensor or dict (NpzFile) of tensors
    """
    # Load
    return np.load(path)
