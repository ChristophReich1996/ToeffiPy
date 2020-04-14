from tqdm import tqdm

import autograd
import autograd.nn as nn
import autograd.nn.functional as F


class NeuralNetwork(nn.Module):
    '''
    This class implements a simple neural network
    '''

    def __init__(self):
        '''
        Constructor
        '''
        # Call super constructor
        super(NeuralNetwork, self).__init__()
        # Init layers and activations
        self.upscale_1 = nn.UpsamplingNearest1d(scale_factor=2)
        self.linear_1 = nn.Linear(in_features=4, out_features=2, bias=True)
        self.acitvation_1 = nn.SeLU()
        self.upscale_2 = nn.UpsamplingNearest1d(scale_factor=2)
        self.conv_2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=3, bias=True)
        self.acitvation_2 = nn.SeLU()
        self.linear_3 = nn.Linear(in_features=2, out_features=2, bias=True)

    def forward(self, input: autograd.Tensor) -> autograd.Tensor:
        '''
        Forward pass
        :param input: (Tensor) Input tensor
        :return: (Tensor) Output tensor
        '''
        # Perform operations
        output = self.upscale_1(input)
        output = self.linear_1(output)
        output = self.acitvation_1(output)
        output = self.upscale_2(output)
        output = self.conv_2(output)
        output = self.acitvation_2(output)
        output = self.linear_3(output)
        return output


if __name__ == '__main__':
    # Init xor data
    input = autograd.Tensor([[[0.0, 0.0]],
                             [[0.0, 1.0]],
                             [[1.0, 0.0]],
                             [[1.0, 1.0]]])

    label = autograd.Tensor([[[0.0, 1.0]],
                             [[1.0, 0.0]],
                             [[1.0, 0.0]],
                             [[0.0, 1.0]]])
    # Init neural network
    neural_network = NeuralNetwork()
    # Print number of model parameters
    print('Model parameter', neural_network.count_params(), flush=True)
    # Init loss function
    loss_function = nn.SoftmaxCrossEntropyLoss(axis=2)
    # Init optimizer
    optimizer = nn.Adam(neural_network.parameters, lr=0.001)
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
    # Neural network into eval mode
    neural_network.eval()
    # Make prediction
    prediction = neural_network(input)
    # Print prediction
    print(round(F.softmax(prediction, axis=2)))
