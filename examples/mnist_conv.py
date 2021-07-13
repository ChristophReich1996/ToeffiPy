from typing import Tuple

from tqdm import tqdm
import numpy as np

import autograd
import autograd.nn as nn
import autograd.nn.functional as F
import autograd.data


class MNIST(autograd.data.Dataset):
    """
    This class implements the mnist dataset
    """

    def __init__(self, path_inputs: str = 'mnist\mnist_small_train_in.txt',
                 path_labels: str = 'mnist\mnist_small_train_out.txt') -> None:
        """
        Constructor method
        :param path_inputs: (str) Path to mnist input data
        :param path_labels: (str) Path to mnist labels
        """
        # Call super constructor
        super(MNIST, self).__init__()
        # Load mnist inputs
        self.inputs = np.genfromtxt(fname=path_inputs, delimiter=',')
        # Reshape inputs
        self.inputs = np.expand_dims(self.inputs, axis=1).reshape((-1, 1, 28, 28))
        # Make autograd tensor
        self.inputs = autograd.Tensor(self.inputs, requires_grad=False)
        # Load mnist labels and convert to one-hot
        self.labels = np.genfromtxt(fname=path_labels, delimiter=',', dtype=int)
        # Convert label to one hot
        self.labels = np.eye(10)[self.labels]
        # Add third dimension to labels
        self.labels = np.expand_dims(self.labels, axis=1)
        # Make autograd tensor
        self.labels = autograd.Tensor(self.labels, requires_grad=False)

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length
        """
        return self.inputs.shape[0]

    def __getitem__(self, item: int) -> Tuple[autograd.Tensor, autograd.Tensor]:
        """
        Method returns the input and the corresponding label
        :param item: (int) Index of sample to be returned
        :return: (Tuple[autograd.Tensor, autograd.Tensor]) Input and corresponding label
        """
        return self.inputs[item], self.labels[item]


class ResidualBlock(nn.Module):
    """
    This class implements a simple residual block consisting of two convolution followed each by an activation function
    , a downsampling operation in the end and a dropout layer.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        """
        super(ResidualBlock, self).__init__()
        # Init main mapping
        self.main_mapping = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                      bias=True),
            nn.BatchNorm2d(num_channels=out_channels, track_running_stats=True),
            nn.PAU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=(1, 1),
                      bias=True),
            nn.BatchNorm2d(num_channels=out_channels, track_running_stats=True),
            nn.PAU(),
            nn.Dropout2D(p=0.1)
        )
        # Init residual mapping
        self.residual_mapping = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1),
                                          bias=True)
        # Init pooling layer
        self.pooling = nn.AvgPool2d(kernel_size=(2, 2))

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        output = self.residual_mapping(input) + self.residual_mapping(input)
        output = self.pooling(output)
        return output


class NeuralNetwork(nn.Module):
    """
    This class implements a simple two layer feed forward neural network for classification.
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        # Call super constructor
        super(NeuralNetwork, self).__init__()
        # Init layers and activations
        self.res_block_1 = ResidualBlock(in_channels=1, out_channels=64)
        self.res_block_2 = ResidualBlock(in_channels=64, out_channels=1)
        self.linear_block = nn.Sequential(
            nn.Linear(in_features=49, out_features=64, bias=True),
            nn.PAU(),
            nn.Linear(in_features=64, out_features=10, bias=False)
        )

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        # Perform operations
        output = self.res_block_1(input)
        output = self.res_block_2(output)
        output = autograd.flatten(output, starting_dim=2)
        output = self.linear_block(output)
        return output


if __name__ == '__main__':
    # Inti training dataset
    dataloader_train = autograd.data.DataLoader(dataset=MNIST(), batch_size=32, shuffle=True)
    # Init neural network
    neural_network = NeuralNetwork()
    # Load pre trained model if needed
    # neural_network.load_state_dict(autograd.load('mnist_conv.npz'))
    # Print number of model parameters
    print('Model parameter', neural_network.count_params(), flush=True)
    # Init loss function
    loss_function = nn.SoftmaxCrossEntropyLoss(axis=2)
    # Init optimizer
    optimizer = nn.Adam(neural_network.parameters, lr=0.001)
    # Neural network into train mode
    neural_network.train()
    # Init number of epochs to perform
    epochs = 5
    # Init progress bar
    progress_bar = tqdm(total=epochs * len(dataloader_train.dataset))
    # Train model
    for epoch in range(epochs):
        for input, label in dataloader_train:
            # Update progress bar
            progress_bar.update(n=input.shape[0])
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
    # Close progress bar
    progress_bar.close()
    # Init test dataset
    dataloader_test = autograd.data.DataLoader(
        dataset=MNIST(path_inputs='mnist\mnist_small_test_in.txt', path_labels='mnist\mnist_small_test_out.txt'),
        batch_size=1)
    # Model into eval mode
    neural_network.eval()
    # Init counter of correctly_classified samples
    correctly_classified = autograd.Tensor(0.0)
    # Test loop to compute tha accuracy
    for input, label in dataloader_test:
        # Make prediction
        prediction = F.softmax(neural_network(input), axis=2)
        # Apply max to get one hot tensor
        prediction = prediction == prediction.max()
        # Compare prediction with label
        correctly_classified += ((prediction == label).sum() == prediction.shape[2])
    # Print accuracy
    print('Accuracy=', str(correctly_classified / len(dataloader_test)), flush=True)
    # Save model
    autograd.save(neural_network.state_dict(), 'mnist_conv.npz')
