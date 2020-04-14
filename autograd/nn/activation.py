import autograd
from autograd import Tensor
from module import Module


class Softplus(Module):
    '''
    This class implements a softplus activation module
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(Softplus, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.softplus(tensor=input)


class ELU(Module):
    '''
    This class implements a elu activation module
    '''

    def __init__(self, alpha: float = 1.0) -> None:
        '''
        Constructor
        :param alpha: (float) Alpha coefficient
        '''
        # Call super constructor
        super(ELU, self).__init__()
        # Save parameter
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.elu(tensor=input, alpha=self.alpha)


class SeLU(Module):
    '''
    This class implements a SeLU activation module
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(SeLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.selu(tensor=input)


class ReLU(Module):
    '''
    This class implements a ReLU activation module
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(ReLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.relu(tensor=input)


class LeakyReLU(Module):
    '''
    This class implements a leaky relu activation module
    '''

    def __init__(self, negative_slope: float = 0.2) -> None:
        '''
        Constructor
        :param negative_slope: (float) Negative slope utilized in leaky relu
        '''
        # Call super constructor
        super(LeakyReLU, self).__init__()
        # Save parameter
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Input tensor
        '''
        return autograd.leaky_relu(tensor=input, negative_slope=self.negative_slope)


class Sigmoid(Module):
    '''
    This class implements a sigmoid activation module
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(Sigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.sigmoid(tensor=input)


class Identity(Module):
    '''
    This class implements an identity mapping
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(Identity, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return input


class Softmax(Module):
    '''
    Class implements the softmax activation module
    '''

    def __init__(self, axis: int = 1) -> None:
        '''
        Constructor
        :param axis: (int) Axis to apply softmax
        '''
        # Call super constructor
        super(Softmax, self).__init__()
        # Save axis argument
        self.axis = axis

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        output_exp = autograd.exp(input)
        output = output_exp / (autograd.sum(output_exp, axis=self.axis, keepdims=True))
        return output


class Tanh(Module):
    '''
    Implementation of the tanh activation module
    '''

    def __init__(self) -> None:
        '''
        Constructor
        '''
        # Call super constructor
        super(Tanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        return autograd.tanh(tensor=input)
