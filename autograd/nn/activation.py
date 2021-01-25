from typing import Optional

import autograd
from . import functional
from autograd import Tensor
from .module import Module
from .parameter import Parameter

import numpy as np


class PAU(Module):
    """
    This class implements the a pade activation unit.
    Source: https://arxiv.org/abs/1907.06732
    """

    def __init__(self, random_init: Optional[bool] = False) -> None:
        """
        Constructor method
        :param random_init: (Optional[bool]) If true parameters are initialized randomly else pau init as leaky ReLU
        """
        # Call super constructor
        super(PAU, self).__init__()
        # Init parameters
        self.m = Parameter(6) if random_init else \
            Parameter(data=np.array([0.02557776, 0.66182815, 1.58182975, 2.94478759, 0.95287794, 0.23319681]))
        self.n = Parameter(5) if random_init else \
            Parameter(data=np.array([0.50962605, 4.18376890, 0.37832090, 0.32407314]))

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Tensor of any shape
        :return: (Tensor) Activated output tensor of the same shape as the input tensor
        """
        return functional.pau(tensor=input, m=self.m, n=self.n)


class Softplus(Module):
    """
    This class implements a softplus activation module
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(Softplus, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.softplus(tensor=input)


class ELU(Module):
    """
    This class implements a elu activation module
    """

    def __init__(self, alpha: Optional[float] = 1.0) -> None:
        """
        Constructor
        :param alpha: (Optional[float]) Alpha coefficient
        """
        # Call super constructor
        super(ELU, self).__init__()
        # Save parameter
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.elu(tensor=input, alpha=self.alpha)


class SeLU(Module):
    """
    This class implements a SeLU activation module
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(SeLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.selu(tensor=input)


class ReLU(Module):
    """
    This class implements a ReLU activation module
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.relu(tensor=input)


class LeakyReLU(Module):
    """
    This class implements a leaky relu activation module
    """

    def __init__(self, negative_slope: Optional[float] = 0.2) -> None:
        """
        Constructor
        :param negative_slope: (Optional[float]) Negative slope utilized in leaky relu
        """
        # Call super constructor
        super(LeakyReLU, self).__init__()
        # Save parameter
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Input tensor
        """
        return autograd.leaky_relu(tensor=input, negative_slope=self.negative_slope)


class Sigmoid(Module):
    """
    This class implements a sigmoid activation module
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.sigmoid(tensor=input)


class Identity(Module):
    """
    This class implements an identity mapping
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return input


class Softmax(Module):
    """
    Class implements the softmax activation module
    """

    def __init__(self, axis: Optional[int] = 1) -> None:
        """
        Constructor
        :param axis: (Optional[float]) Axis to apply softmax
        """
        # Call super constructor
        super(Softmax, self).__init__()
        # Save axis argument
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        output_exp = autograd.exp(input)
        output = output_exp / (autograd.sum(output_exp, axis=self.axis, keepdims=True))
        return output


class Tanh(Module):
    """
    Implementation of the tanh activation module
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        return autograd.tanh(tensor=input)
