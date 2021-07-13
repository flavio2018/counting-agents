import torch

from torch import nn

from src.utils import initialize_weights
from src.Reward import Reward
from src.SingleAgentEnv import SingleAgentEnv


class MLPAgent(nn.Module):

    def __init__(self, input_dim, n_layers, vis_rep_size = None, action_space_size = None):
        super().__init__()

        self.input_dim = input_dim
        self.n_layers = n_layers
        self.vis_rep_size = vis_rep_size
        self.action_space_size = action_space_size

        size_flattened_rep = self.input_dim * self.input_dim * self.n_layers

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
        leaky_relu = nn.LeakyReLU()

        x = self.flattener(x)
        x = self.linear1(x)
        x = leaky_relu(x)

        if self.vis_rep_size != None:
            x = self.linear2(x)
            x = leaky_relu(x)

        return x


if __name__ == '__main__':
    reward = Reward(**{'bad_label_punishment': False, 'curiosity': False, 'time_penalty': False})
    env = SingleAgentEnv(reward, 3, 1, 4, 8, 1, 10, 1, 4)
    agent_params = {
        'input_dim': env.obs_dim,
        'n_layers': 4,
        'vis_rep_size': 200,
        'action_space_size': 11,
    }
    agent = MLPAgent(**agent_params)

    print(agent(env.reset()))