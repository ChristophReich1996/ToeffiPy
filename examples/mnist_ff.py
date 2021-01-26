from typing import Tuple

from tqdm import tqdm
import numpy as np

import autograd
import autograd.nn as nn
import autograd.nn.functional as F
import autograd.data


class MNIST(autograd.data.Dataset):

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
        self.inputs = np.expand_dims(self.inputs, axis=1)
        self.inputs = autograd.Tensor(self.inputs, requires_grad=False)
        # Load mnist labels and convert to one-hot
        self.labels = np.genfromtxt(fname=path_labels, delimiter=',', dtype=int)
        self.labels = np.eye(10)[self.labels]
        self.labels = np.expand_dims(self.labels, axis=1)
        self.labels = autograd.Tensor(self.labels, requires_grad=False)

    def __len__(self) -> int:
        """
        Method returns the length of the dataset
        :return: (int) Length
        """
        return self.inputs.shape[0]

    def __getitem__(self, item) -> Tuple[autograd.Tensor, autograd.Tensor]:
        """
        Method returns the input and the corresponding label
        :param item:
        :return:
        """
        return self.inputs[item], self.labels[item]


class NeuralNetwork(nn.Module):
    """
    This class implements a simple two layer feed forward neural network for classification.
    """

    def __init__(self):
        """
        Constructor
        """
        # Call super constructor
        super(NeuralNetwork, self).__init__()
        # Init layers and activations
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, bias=True),
            nn.SeLU(),
            nn.Dropout(p=0.1),
            nn.Linear(in_features=28 * 28 - 4, out_features=64, bias=True),
            nn.BatchNorm1d(num_channels=1),
            nn.SeLU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Linear(in_features=32, out_features=32, bias=True),
            nn.SeLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Linear(in_features=16, out_features=10, bias=True),
        )

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        """
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        """
        # Perform operations
        output = self.layers(input)
        return output


if __name__ == '__main__':
    # Inti training dataset
    dataloader_train = autograd.data.DataLoader(dataset=MNIST(), batch_size=128, shuffle=True)
    # Init neural network
    neural_network = NeuralNetwork()
    # Print number of model parameters
    print('Model parameter', neural_network.count_params(), flush=True)
    # Init loss function
    loss_function = nn.SoftmaxCrossEntropyLoss(axis=2)
    # Init optimizer
    optimizer = nn.RMSprop(neural_network.parameters, lr=0.001)
    # Neural network into train mode
    neural_network.train()
    # Init number of epochs to perform
    epochs = 2
    # Init progress bar
    progress_bar = tqdm(total=epochs * len(dataloader_train.dataset))
    # Train nn
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
    # Test model
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
