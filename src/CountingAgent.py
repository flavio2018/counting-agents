"""
This script contains the class of the complete agent involved in the communication/counting task. The agent is composed of a core ConvLSTM module and two Fully Connected layers which build a visual representation and it's mapping to the actions space, respectively.
"""

import torch
from torch import nn
import numpy as np
from ConvLSTMCell import ConvLSTMCell
from utils import initialize_weights

class CountingAgent(nn.Module):
    """
    Args:
        input_channels: The number of channels in the input data.
        input_dim: The length of of side of input data. Data is
            presumed to have identical width and heigth.
        n_kernels: The number of kernels of Conv2d layers.
        dropout: The value of p parameter for dropout layers.
        pool_kernel_size: Size of the window the pooling filter acts on.
        vis_rep_size: Size of visual representation (should be 10*no_action?)
        action_space_size: Number of actions the agent can take.
    """
    def __init__(self, input_channels, input_dim, n_kernels, dropout, pool_kernel_size, vis_rep_size=30, action_space_size=3):
        
        super().__init__()
        self.input_channels = input_channels
        self.input_dim = input_dim
        self.n_kernels = n_kernels
        self.dropout = dropout
        self.vis_rep_size = vis_rep_size
        self.action_space_size = action_space_size
        
        # Core ConvLSTM module
        self.ConvLSTMCell = ConvLSTMCell(
            input_channels=self.input_channels,
            input_dim=self.input_dim,
            dropout=self.dropout,
            n_kernels=self.n_kernels,
            batch_norm=False
        )
        
        size_flattened_rep = self.input_dim * self.input_dim * self.n_kernels
        
        # From flattened-2D to visual representation
        # From visual representation to action space
        self.Vis2Act = nn.Sequential(
            torch.nn.Flatten(start_dim=1), # flattens all dimensions (keeps batches)
            nn.Linear(
                in_features=size_flattened_rep,
                out_features=self.vis_rep_size
            ),
            nn.Linear(
                in_features=self.vis_rep_size,
                out_features=self.action_space_size
            ),
            nn.Sigmoid()
        )
        
        self.apply(initialize_weights)
        
    def forward(self, x):
        x, C = self.ConvLSTMCell(x)
        
        x = self.Vis2Act(x)
        
        return x
