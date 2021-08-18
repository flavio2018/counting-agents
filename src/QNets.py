'''
This code is taken from: https://github.com/dxyang/DQN_pytorch/blob/master/model.py
'''

import torch
import torch.nn as nn
import numpy as np

class FC(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(FC, self).__init__()
        input_dim = in_channels*4*4
        self.fc3 = nn.Linear(in_features=input_dim, out_features=num_actions)
        self.fc3.weight.requires_grad = False
        self.relu = nn.ReLU()

    def forward(self, x):
        # simple version
        x = x.view(x.size(0), -1)
        x = self.fc3(x)
        return x

class CNN(nn.Module):
    def __init__(self, in_channels, num_actions, example_input=None, dim=2):
        super(CNN, self).__init__()

        if(dim == 2):
            kernel_size = (3,3)
            self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=kernel_size, stride=1, padding=1)
            #self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=kernel_size, stride=1)
        else:
            kernel_size = 3
            self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=2, kernel_size=kernel_size, stride=1, padding=1)
            #self.conv2 = nn.Conv1d(in_channels=2, out_channels=4, kernel_size=kernel_size, stride=1)

        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        if(example_input is not None):
            inpy = torch.from_numpy(example_input).float().unsqueeze(0)
            if(dim==1):
                inpy = inpy.squeeze(-1)
            x = self.conv1(inpy)
            #x = self.conv2(x)
            input_dim = x.size().numel()
        else:
            input_dim = in_channels * 4 * 4
        #input_dim = conv_output_shape((3,3), kernel_size=1, stride=1, pad=0, dilation=1)
        self.fc1 = nn.Linear(in_features=input_dim, out_features=512)  #512
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)
        self.fc3 = nn.Linear(in_features=input_dim, out_features=num_actions)

        self.relu = nn.ReLU()

        self.dim = dim

    def forward(self, x):
        if(self.dim == 1):
            x = x.squeeze(3)
        x = self.relu(self.conv1(x))
        #x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # simple version
        #x = x.view(x.size(0), -1)
        #x = self.fc3(x)
        return x

class N_Concat_CNNs(nn.Module):
    def __init__(self, in_channels, num_actions, shared_policy=False, example_input=None, dim=2):
        super(N_Concat_CNNs, self).__init__()
        #def __init__(self):
        example_input_ = example_input[0] if dim==2 else example_input[0].squeeze(2)
        self.CNN_1 = CNN(in_channels, num_actions, example_input=example_input_, dim=dim)
        #self.CNN_2 = CNN(in_channels, num_actions)
        #for params in self.CNN_2.parameters():
        #    params.requires_grad = False
        self.shared_policy = shared_policy
        self.dim = dim


    def forward(self, x_list):
        #out1 = self.CNN_1(x_list[:, 0, :, :, :])
        #out2 = self.CNN_1(x_list[:, 1, :, :, :])
        input1 = x_list[:, 0, :, :, :]
        input2 = x_list[:, 1, :, :, :]
        out1 = self.CNN_1(input1)
        out2 = self.CNN_1(input2)
        #if(self.shared_policy):
        #    out2 = self.CNN_1(x_list[:, 1, :, :, :])
        #else:
        #    out2 = self.CNN_2(x_list[:, 1, :, :, :])

        return [out1, out2]

class Dueling_DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(Dueling_DQN, self).__init__()
        self.num_actions = num_actions

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1_adv = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc1_val = nn.Linear(in_features=7 * 7 * 64, out_features=512)

        self.fc2_adv = nn.Linear(in_features=512, out_features=num_actions)
        self.fc2_val = nn.Linear(in_features=512, out_features=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        batch_size = x.size(0)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        adv = self.relu(self.fc1_adv(x))
        val = self.relu(self.fc1_val(x))

        adv = self.fc2_adv(adv)
        val = self.fc2_val(val).expand(x.size(0), self.num_actions)

        x = val + adv - adv.mean(1).unsqueeze(1).expand(x.size(0), self.num_actions)
        return x


def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    from math import floor
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, kernel_size)
    h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
    w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
    return h, w