# ToeffiPy

ToeffiPy is a fully [NumPy](https://numpy.org/) based autograd and deep learning library. The core tensor class of 
ToeffiPy is highly inspired by the 
[live coding challenge](https://www.youtube.com/watch?v=RxmBukb-Om4&list=PLeDtc0GP5ICldMkRg-DkhpFX1rRBNHTCs) 
of [Joel Grus](https://github.com/joelgrus/). His code can be found in [this](https://github.com/joelgrus/autograd/) 
repository. The user experience (nn.Module etc.) of ToeffiPy should be similar to [PyTorch](https://pytorch.org/). 
The main purpose of this library is educational. ToeffiPy should give an inside about how a modern autograd/deep 
learning library works. To implement operations like a convolution as low level as possible, but also efficient, NumPy 
was chosen. Since NumPy is highly optimized, ToeffiPy is suitable for small machine learning projects.

Example use:
```python
import autograd
import numpy as np
import matplotlib.pyplot as plt

x = autograd.Tensor(np.linspace(-4, 4, 1000), requires_grad=True)
y = autograd.tanh(x)
y.sum().backward() # Compute gradients
plt.plot(x.data, y.data)
plt.plot(x.data, x.grad.data)
plt.show()
```

![Tanh plot](examples/tanh.png)

# Theory

ToeffiPy can handle Python features like loops or if statements. This is due to the fact that ToeffiPy utilized a 
dynamic backward graph. To compute the gradients of a graph simple
[backpropagation](https://en.wikipedia.org/wiki/Backpropagation) is used. However, **only first-order derivatives are 
supported**.

# Usage


# Examples

* [XOR problem](examples/xor.py)
* [Simple linear regression](examples/regression.py)
* [Simple regression with a neural network](examples/regression_nn.py)
* [MNIST classification with a feed forward neural network](examples/mnist_ff.py)
* [MNIST classification with a CNN](examples/mnist_conv.py)

# Installation

