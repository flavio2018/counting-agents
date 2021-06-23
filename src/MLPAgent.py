import torch

from torch import nn

from src.utils import initialize_weights


class MLPAgent(nn.Module):

    def __init__(self, **kwargs):
        __slots__ = ('input_dim', 'n_layers', 'vis_rep_size', 'action_space_size')
        super().__init__()

        for attribute in __slots__:
            if attribute in kwargs:
                setattr(self, attribute, kwargs[attribute])
            else:
                setattr(self, attribute, None)

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