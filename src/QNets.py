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
    def __init__(self, in_channels, num_actions):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, stride=1)
        #self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(in_features=16, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        input_dim = in_channels*4*4
        self.fc3 = nn.Linear(in_features=input_dim, out_features=num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        # simple version
        #x = x.view(x.size(0), -1)
        #x = self.fc3(x)
        return x

class N_Concat_CNNs(nn.Module):
    def __init__(self, in_channels, num_actions, shared_policy=False):
        super(N_Concat_CNNs, self).__init__()
        self.CNN_1 = CNN(in_channels, num_actions)
        #self.CNN_2 = CNN(in_channels, num_actions)
        #for params in self.CNN_2.parameters():
        #    params.requires_grad = False
        self.shared_policy = shared_policy


    def forward(self, x_list):
        #out1 = self.CNN_1(x_list[:, 0, :, :, :])
        #out2 = self.CNN_1(x_list[:, 1, :, :, :])
        out1 = self.CNN_1(x_list[:, 0, :, :, :])
        out2 = self.CNN_1(x_list[:, 1, :, :, :])
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
