import torch

from torch import nn

from src.utils import initialize_weights


class CNNAgent(nn.Module):

    def __init__(self, input_dim: int, input_channels: int, n_kernels: int,
                 vis_rep_size: int = None, action_space_size: int = None):
        super().__init__()

        self.input_dim = input_dim
        self.input_channels = input_channels
        self.n_kernels = n_kernels
        self.vis_rep_size = vis_rep_size
        self.action_space_size = action_space_size

        size_flattened_rep = self.input_dim * self.input_dim * self.n_kernels

        self.kernel_size = 3
        self.padding = 1
        self.ConvLayer = nn.Conv2d(in_channels=self.input_channels,
                                   out_channels=self.n_kernels,
                                   kernel_size=self.kernel_size,
                                   padding=self.padding,
                                   bias=True)

        # From flattened-2D to visual representation
        # From visual representation to action space
        self.flattener = torch.nn.Flatten(start_dim=1)

        if vis_rep_size != None:
            self.linear1 = nn.Linear(
                in_features=size_flattened_rep,
                out_features=self.vis_rep_size
            )
            self.linear2 = nn.Linear(
                in_features=self.vis_rep_size,
                out_features=self.action_space_size
            )
        else:
            self.linear1 = nn.Linear(
                in_features=size_flattened_rep,
                out_features=self.action_space_size
            )

        self.apply(initialize_weights)

    def forward(self, x):
        sigm = nn.Sigmoid()
        x = self.ConvLayer(x)

        x = self.flattener(x)
        x = self.linear1(x)
        x = sigm(x)

        if self.vis_rep_size != None:
            x = self.linear2(x)
            x = sigm(x)

        return x

