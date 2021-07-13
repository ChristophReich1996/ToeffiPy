from typing import Iterator, Callable

import numpy as np

from .parameter import Parameter


class Optimizer(object):
    """
    Super class of optimizer
    """

    def __init__(self, parameters: Callable[[], Iterator[Parameter]]) -> None:
        """
        Constructor method
        :param parameters: (Callable[[], Iterator[Parameter]]) Callable function which returns the parameter of a module
        """
        self.parameters = parameters

    def step(self) -> None:
        """
        Optimization step method
        """
        raise NotImplementedError()


class SGD(Optimizer):
    """
    Class implements a stochastic gradient decent optimizer
    """

    def __init__(self, parameters: Callable[[], Iterator[Parameter]], lr: float = .01) -> None:
        """
        Constructor
        :param parameters: (Iterator[Parameter]) Module parameters to be optimized
        :param lr: (float) Learning rate to be utilized
        """
        # Call super constructor
        super(SGD, self).__init__(parameters=parameters)
        # Save learning rate
        self.lr = lr

    def step(self) -> None:
        """
        Method performs optimization step
        """
        # Loop over all parameters
        for parameter in self.parameters():
            # Perform gradient decent
            parameter -= self.lr * parameter.grad.data


class SGDMomentum(Optimizer):

    def __init__(self, parameters: Callable[[], Iterator[Parameter]], lr: float = .01, momentum: float = .9) -> None:
        """
        Constructor
        :param parameters: (Iterator[Parameter]) Module parameters to be optimized
        :param lr: (float) Learning rate to be utilized
        :param momentum: (float) Momentum factor
        """
        # Call super constructor
        super(SGDMomentum, self).__init__(parameters=parameters)
        # Save parameters
        self.lr = lr
        self.momentum = momentum
        # Init dict to store momentum tensors
        self.velocity = dict()
        # Init counter
        self.t = 0

    def step(self) -> None:
        """
        Method performs optimization step
        """
        counter_parameter = 0
        for parameter in self.parameters():
            # Init average squared grad tensor
            if self.t == 0:
                self.velocity[str(counter_parameter)] = np.zeros_like(parameter.data)
            # Calc current velocity by running average
            self.velocity[str(counter_parameter)] = \
                self.momentum * self.velocity[str(counter_parameter)] - (1.0 - self.momentum) * parameter.grad.data
            # Perform optimization
            parameter += self.lr * self.velocity[str(counter_parameter)]
            # Increment parameter counter
            counter_parameter += 1
            # Increment counter
        self.t += 1


class Adam(Optimizer):

    def __init__(self, parameters: Callable[[], Iterator[Parameter]], lr: float = .001, beta_1: float = .9,
                 beta_2: float = .999, eps: float = 1e-08) -> None:
        """
        Constructor
        :param parameters: (Iterator[Parameter]) Module parameters to be optimized
        :param lr: (float) Learning rate to be utilized
        :param beta_1: (float) Coefficient for running first oder average
        :param beta_2: (float) Coefficient for running second oder average
        :param eps: (float) Constant for numerical stability
        """
        # Call super constructor
        super(Adam, self).__init__(parameters=parameters)
        # Save parameters
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        # Init counter
        self.t = 0
        # Init dicts to store moving averages
        self.first = dict()
        self.second = dict()

    def step(self) -> None:
        """
        Method performs optimization step
        """
        # Loop over all parameters
        counter_parameter = 0
        for parameter in self.parameters():
            # Init fist and second order momentum
            if self.t == 0:
                self.first[str(counter_parameter)] = np.zeros_like(parameter.data)
                self.second[str(counter_parameter)] = np.zeros_like(parameter.data)
            # Calc moving averages
            self.first[str(counter_parameter)] = \
                (1.0 - self.beta_1) * parameter.grad.data + self.beta_1 * self.first[str(counter_parameter)]
            self.second[str(counter_parameter)] = \
                (1.0 - self.beta_2) * (parameter.grad.data ** 2) + self.beta_2 * self.second[str(counter_parameter)]
            # Calc corrections
            fist = self.first[str(counter_parameter)] / (1 - self.beta_1 ** (self.t + 1))
            second = self.second[str(counter_parameter)] / (1 - self.beta_2 ** (self.t + 1))
            # Perform optimization
            parameter -= self.lr * fist / (np.sqrt(second) + self.eps)
            # Increment parameter counter
            counter_parameter += 1
        # Increment counter
        self.t += 1


class RMSprop(Optimizer):
    """
    Root mean squared prop optimizer implementation
    """

    def __init__(self, parameters: Callable[[], Iterator[Parameter]], lr: float = .01, alpha: float = .99,
                 eps=1e-08) -> None:
        """
        Constructor
        :param parameters: (Iterator[Parameter]) Module parameters to be optimized
        :param lr: (float) Learning rate to be utilized
        :param alpha: (float) Smoothing constant
        :param eps: (float) Constant for numerical stability
        """
        # Call super constructor
        super(RMSprop, self).__init__(parameters=parameters)
        # Save parameters
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        # Init counter
        self.t = 0
        # Init dict to store average squared grad
        self.avg_squared_grad = dict()

    def step(self) -> None:
        """
        Method performs optimization step
        """
        # Loop over all parameters
        counter_parameter = 0
        for parameter in self.parameters():
            # Init average squared grad tensor
            if self.t == 0:
                self.avg_squared_grad[str(counter_parameter)] = np.zeros_like(parameter.data)
            # Calc average
            self.avg_squared_grad[str(counter_parameter)] = \
                self.avg_squared_grad[str(counter_parameter)] * self.alpha + parameter.grad.data ** 2 * (1 - self.alpha)
            # Perform optimization
            parameter -= self.lr * parameter.grad.data / \
                         (np.sqrt(self.avg_squared_grad[str(counter_parameter)]) + self.eps)
            # Increment parameter counter
            counter_parameter += 1
            # Increment counter
        self.t += 1
