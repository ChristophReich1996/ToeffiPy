from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

import autograd
import autograd.nn as nn


class NeuralNetwork(nn.Module):
    """
    This class implements a simple neural network
    """

    def __init__(self) -> None:
        # Call super constructor
        super(NeuralNetwork, self).__init__()
        # Init modules
        self.modules = nn.Sequential(
            nn.Linear(in_features=1, out_features=16, bias=True),
            nn.PAU(),
            nn.Linear(in_features=16, out_features=16, bias=True),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(in_features=16, out_features=16, bias=True),
            nn.LeakyReLU(negative_slope=0.5),
            nn.Linear(in_features=16, out_features=1, bias=True)
        )

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        # Perform operations
        output = self.modules(input)
        return output


if __name__ == '__main__':
    # Init sample size
    sample_size = 10
    # Make data
    x = np.random.uniform(-10, 10, (sample_size))
    y = 2.5 * x ** 3 - 0.5 * x + np.random.uniform(0, 0.25, (sample_size))
    # Make input and label
    input = autograd.Tensor(x[:, None, None])
    label = autograd.Tensor(y[:, None, None])
    # Init neural network
    neural_network = NeuralNetwork()
    # Init loss function
    loss_function = nn.L1Loss()
    # Init optimizer
    optimizer = nn.Adam(neural_network.parameters, lr=0.003)
    # Neural network into train mode
    neural_network.train()
    # Init progress bar
    progress_bar = tqdm(total=10000)
    # Train nn
    for _ in range(10000):
        # Update progress bar
        progress_bar.update(n=1)
        # Reset gradients of neural network
        neural_network.zero_grad()
        # Make prediction
        prediction = neural_network(input)
        # Calc loss
        loss = loss_function(prediction, label)
        # Calc gradients
        loss.backward()
        # Perform optimization step
        optimizer.step()
        # Show loss in progress bar
        progress_bar.set_description('Loss={:.5f}'.format(loss.data))
    # Plot prediction and label
    plt.scatter(x, y)
    x = np.linspace(-10, 10, (1000))
    input = autograd.Tensor(x[:, None, None])
    prediction = neural_network(input)
    plt.plot(x, prediction.data[:, 0, 0])
    plt.show()
