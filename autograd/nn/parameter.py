import numpy as np

from autograd.tensor import Tensor


class Parameter(Tensor):
    '''
    Implementation of a nn Parameter which always requires grad.
    '''

    def __init__(self, *shape: int, data: np.ndarray = None) -> None:
        '''
        Constructor method
        :param shape: (int) Dimensions of the parameter data
        '''
        if data is None:
            data = 0.1 * np.random.randn(*shape)
            super(Parameter, self).__init__(data=data, requires_grad=True)
        else:
            super(Parameter, self).__init__(data=data, requires_grad=True)
