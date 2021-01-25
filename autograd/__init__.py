# Init tensor object
from .tensor import Tensor
# Init operations as functions
from .tensor import add, sub, mul, matmul, pow, div, abs, mean, sum, var, std, unsqueeze, squeeze, clone, stack, \
    pad_1d, pad_2d, flatten, save, load
# Init functions
from .function import tanh, sigmoid, relu, leaky_relu, elu, selu, sin, cos, tan, sqrt, exp, log, softplus
