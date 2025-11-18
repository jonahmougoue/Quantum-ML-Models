import math
import torch
from torch import nn
from torch import Tensor

class EnergyCNN(nn.Module):
    def __init__(self, cnn_channels:list,hidden_layers:list=[]):
        """
        Model for predicting energy of a wavefunction using a potential map
        :param cnn_channels: Tuple of tuples containing parameters for each CNN layer
        :param hidden_layers: Tuple containing ints describing output for each hidden layer
        """
        super().__init__()
        stride = 2
        padding = 1
        cnn_layers = []
        if cnn_channels:
            cnn_layers.append(nn.Conv2d(1,cnn_channels[0][0],kernel_size=cnn_channels[0][1],stride=cnn_channels[0][2],padding=cnn_channels[0][3],padding_mode='replicate'))
            cnn_layers.append(nn.ReLU())
            for i in range(1,len(cnn_channels)):
                cnn_layers.append(nn.Conv2d(cnn_channels[i-1][0],cnn_channels[i][0],kernel_size=cnn_channels[i][1],stride=cnn_channels[i][2],padding=cnn_channels[i][3],padding_mode='replicate'))
                cnn_layers.append(nn.ReLU())

        self.cnn_stack = nn.Sequential(*cnn_layers)
        cnn_output = self.cnn_stack(torch.zeros(1,1,256,256))

        linear_layers = []
        if hidden_layers:
            linear_layers.append(nn.Flatten())
            linear_layers.append(nn.Linear(cnn_output.numel(),hidden_layers[0]))
            linear_layers.append(nn.ReLU())
            for i in range(len(hidden_layers)-1):
                linear_layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
                linear_layers.append(nn.ReLU())
            linear_layers.append(nn.Linear(hidden_layers[-1], 1))
        else:
            linear_layers.append(nn.Linear(math.prod(cnn_output),1))
        self.linear_stack = nn.Sequential(*linear_layers)

    def forward(self,X:Tensor)->Tensor:
        """
        Predicts energy for the wavefunction
        :param X: Potential image
        :return: Energy prediction
        """
        X = self.linear_stack(self.cnn_stack(X))
        return X