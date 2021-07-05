import torch

from torch import nn

from src.utils import initialize_weights


class MLPAgent(nn.Module):

    def __init__(self, input_dim, n_layers, vis_rep_size, action_space_size):

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.vis_rep_size = vis_rep_size
        self.action_space_size = action_space_size

        size_flattened_rep = self.input_dim * self.input_dim * self.n_layers

        # From flattened-2D to visual representation
        # From visual representation to action space
        self.Vis2Act = nn.Sequential(
            torch.nn.Flatten(start_dim=1),  # flattens all dimensions (keeps batches)
            nn.Linear(
                in_features=size_flattened_rep,
                out_features=self.action_space_size
            ),
            nn.Sigmoid()
        )

        self.apply(initialize_weights)

    def forward(self, x):
        return self.Vis2Act(x)