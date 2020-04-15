from typing import Iterator, Any, Dict, Union

import inspect
import numpy as np

from autograd.tensor import Tensor
from parameter import Parameter


class Module(object):
    '''
    Super class of an autograd module
    '''

    def __init__(self) -> None:
        '''
        Constructor method
        '''
        # Init train mode flag
        self.train_mode = True

    def __call__(self, *args: Any) -> Tensor:
        '''
        Link forward method to __call___
        :param args: (Tensor or different object) Inputs
        :return: (Tensor) Outputs
        '''
        # Make clone of input tensors
        args = list(args)
        for index, arg in enumerate(args):
            if isinstance(arg, Tensor):
                args[index] = arg.clone()
        args = tuple(args)
        # Make copy of inputs
        return self.forward(*args)

    def forward(self, *input: Any) -> Tensor:
        '''
        Forward method to be implemented in children class
        :param input: (Tensor or different object) Inputs
        :return: (Tensor) Outputs
        '''
        raise NotImplementedError()

    def parameters(self) -> Iterator[Parameter]:
        '''
        Method returns all parameters included in the module
        :return: (Iterator[Parameter]) Iterator including all parameters
        '''
        # Iterate over all members of the module
        for _, value in inspect.getmembers(self):
            # Case if object is a nn parameter
            if isinstance(value, Parameter):
                yield value
            # Case if object is module
            elif isinstance(value, Module):
                # Call parameters method again
                yield from value.parameters()

    def state_dict(self, return_data: bool = True) -> Dict[str, Union[Tensor, np.ndarray]]:
        # Init state dict
        state_dict = dict()
        # Get parameters
        for parameter in self.parameters():
            # Put parameter into dict
            if return_data:
                state_dict[str(len(state_dict))] = parameter.data
            else:
                state_dict[str(len(state_dict))] = parameter
        return state_dict

    def load_state_dict(self, state_dict: np.lib.npyio.NpzFile) -> None:
        '''
        Function loads a state dict
        :param state_dict: (np.lib.npyio.NpzFile) State dict as a NpzFile to be loaded
        '''
        # Loop over state dict and parameters
        for name, parameter in zip(state_dict.files, self.parameters()):
            # Check sizes
            assert state_dict[name].shape == parameter.shape, 'Error while loading state dict.'
            # Set data
            parameter.data = state_dict[name]

    def zero_grad(self) -> None:
        '''
        Method zeros gradient of all parameters
        '''
        # Iterate over all parameters
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self) -> None:
        '''
        Method sets train flag of all modules to true.
        '''
        # Set own flag
        self.train_mode = True
        # Iterate over all members of the module
        for _, value in inspect.getmembers(self):
            if isinstance(value, Module):
                # Call recursively train function again
                value.train()

    def eval(self) -> None:
        '''
        Method sets train flag of all modules to false.
        '''
        # Set own flag
        self.train_mode = False
        # Iterate over all members of the module
        for _, value in inspect.getmembers(self):
            if isinstance(value, Module):
                # Set train flag
                value.train_mode = False
                # Call recursively train function again
                value.eval()

    def count_params(self) -> int:
        '''
        Method returns the number of learnable parameters present in the module.
        :return: (int) Number of learnable parameters
        '''
        num_parameters = 0
        for parameter in self.parameters():
            num_parameters += parameter.data.size
        return num_parameters


class Sequential(Module):
    '''
    Sequential model class
    '''

    def __init__(self, *modules: Module) -> None:
        '''
        Constructor
        :param modules: (Module) Different number of layers
        '''
        # Call super constructor
        super(Sequential, self).__init__()
        # Save modules
        self.modules = modules

    def parameters(self) -> Iterator[Parameter]:
        '''
        Method returns all parameters included in the module
        :return: (Iterator[Parameter]) Iterator including all parameters
        '''
        # Iterate over all members of the module
        for _, value in inspect.getmembers(self):
            # Case of module tuple
            if isinstance(value, tuple):
                # Iterate over all elements in tuple
                for item in value:
                    # If instance is a module yield parameters
                    if isinstance(item, Module):
                        yield from item.parameters()

    def forward(self, input: Tensor) -> Tensor:
        '''
        Forward method
        :param input: (Tensor) Inputs
        :return: (Tensor) Outputs
        '''
        output = input
        for layer in self.modules:
            output = layer(output)
        return output
