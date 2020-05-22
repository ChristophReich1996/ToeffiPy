from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import autograd
import autograd.nn as nn


class Linear(nn.Module):
    """
    Linear module
    """

    def __init__(self) -> None:
        # Call super constructor
        super(Linear, self).__init__()
        # Init modules
        self.linear = nn.Linear(in_features=4, out_features=1)

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        # Perform operations
        output = self.linear(input)
        return output


if __name__ == '__main__':
    # Init sample size
    sample_size = 10
    # Make data
    x = np.random.uniform(-2, 2, (sample_size))
    y = 2.5 * x ** 3 - 0.5 * x + np.random.uniform(0, 0.25, (sample_size))
    # Make input and label
    input = autograd.Tensor(np.array([np.ones_like(x), x, x ** 2, x ** 3]).transpose((1, 0)))
    label = autograd.Tensor(y[:, None])
    # Init neural network
    linear = Linear()
    # Init loss function
    loss_function = nn.MSELoss()
    # Init optimizer
    optimizer = nn.SGD(linear.parameters, lr=0.001)
    # Neural network into train mode
    linear.train()
    # Init progress bar
    progress_bar = tqdm(total=10000)
    # Train nn
    for _ in range(10000):
        # Update progress bar
        progress_bar.update(n=1)
        # Reset gradients of neural network
        linear.zero_grad()
        # Make prediction
        prediction = linear(input)
        # Calc loss
        loss = loss_function(prediction, label)
        # Calc gradients
        loss.backward()
        # Perform optimization step
        optimizer.step()
        # Show loss in progress bar
        progress_bar.set_description('Loss={:.5f}'.format(loss.data))
    # Plot training data
    plt.scatter(x, y)
    # Make validation data
    x = np.linspace(-2, 2, (1000))
    input = autograd.Tensor(np.array([np.ones_like(x), x, x ** 2, x ** 3]).transpose((1, 0)))
    # Model in eval mode
    linear.eval()
    # Predict
    prediction = linear(input)
    # Plot prediction
    plt.plot(x, prediction.data[:, 0])
    plt.show()
