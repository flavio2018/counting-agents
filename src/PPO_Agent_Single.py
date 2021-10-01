# Taken from:
# https://github.com/nikhilbarhate99/PPO-PyTorch
############################### Import libraries ###############################


import os
import glob
import time
from datetime import datetime

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np
import pytorch_utils as ptu
from ReplayMemory import ReplayMemory, NaivePrioritizedBuffer
#from QNets import N_Concat_CNNs, CNN
from ActorCriticNets import CNN, Actor_CNN, Actor_FC, Critic_FC

################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []


    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init, env):
        super(ActorCritic, self).__init__()

        self.params = env.params

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)


        # Observation and action sizes for individual agents
        self.ob_dim  = env.state.shape  # Assumes that all agents have same state-dim as agent[0]
        self.ext_shape = env.ext_shape
        dimmy = 1 if self.ext_shape[1] == 1 else 2

        print("ob_dim: ", self.ob_dim)
        n_channels, screen_height, screen_width = self.ob_dim
        self.ac_dim = env.action.shape
        self.n_actions = self.ac_dim[0]

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            if(self.params['net_type'] == 'CNN'):
                self.actor = Actor_CNN(n_channels, self.n_actions, example_input=env.state, dim=dimmy)
            elif (self.params['net_type'] == 'FC'):
                self.actor = Actor_FC(state_dim, action_dim)
            '''
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )
            '''


        # critic
        if (self.params['net_type'] == 'CNN'):
            self.critic = CNN(n_channels, num_actions=1, example_input=env.state, dim=dimmy)
        elif (self.params['net_type'] == 'FC'):
            self.critic = Critic_FC(state_dim)
        '''
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )
        '''

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError


    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()


    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # for single action continuous environments
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            #state = preprocess_input(self.params, state)
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO_Agent_Single:
    def __init__(self, env, agent_params):

        self.params = agent_params
        self.env = env
        self.eps = None
        #state_dim = self.env.state.shape
        state_dim = torch.from_numpy(self.env.state).flatten().shape[0]
        action_dim = self.ac_dim = self.n_actions = self.env.action.shape[0]

        # Observation and action sizes for individual agents
        self.ob_dim  = self.env.state.shape  # Assumes that all agents have same state-dim as agent[0]
        self.ext_shape = self.env.ext_shape
        dimmy = 1 if self.ext_shape[1] == 1 else 2
        self.dimmy = dimmy

        has_continuous_action_space = False
        self.has_continuous_action_space = has_continuous_action_space
        action_std_init = 0.6
        K_epochs = 40  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        gamma = 0.99  # discount factor

        lr_actor = 0.0003  # learning rate for actor network
        lr_critic = 0.001  # learning rate for critic network

        random_seed = 0  # set random seed if required (0 = no random seed)

        if has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.env).to(device)
        self.policy_net = self.policy.actor
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init, self.env).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

        # Memory
        if self.params['PrioratizedReplayMemory']:
            self.memory = NaivePrioritizedBuffer(self.params['MEMORY_CAPACITY'])
        else:
            self.memory = ReplayMemory(self.params['MEMORY_CAPACITY'])


    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")


    def select_action(self, state, i_episode=0, collect=False, deterministic=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = preprocess_input(self.params, state)
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.logprobs.append(action_logprob)

            return action.item()


    def optimize_model(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)


        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.05*dist_entropy  # original beta: 0.01

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

        return loss.mean()


    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)


    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    def convert_to_memory_compatible_format(self, state, next_state, action, reward):
        if next_state is not None:
            next_state = ptu.from_numpy(next_state)
        #if (state is numpy_array):
        if(isinstance(state, (np.ndarray, np.generic))):
            state = ptu.from_numpy(state)
        state_for_memory = state.unsqueeze(0)
        next_state_for_memory = next_state.unsqueeze(0)
        action_for_memory = ptu.from_numpy(np.array(action)).unsqueeze(0).unsqueeze(1).type(torch.int64)
        reward = torch.tensor([ptu.from_numpy(np.array(reward))])

        return state_for_memory, next_state_for_memory, action_for_memory, reward

    def init_device(self):
        if torch.cuda.is_available() and self.agent_params['use_gpu']:
            self.device = torch.device("cuda:" + str(self.agent_params['gpu_id']))
            print("Using GPU id {}".format(self.agent_params['gpu_id']))
        else:
            self.device = torch.device("cpu")
            print("GPU not detected. Defaulting to CPU.")


def preprocess_input(params, state):
    if (isinstance(state, (np.ndarray, np.generic))):
        state = ptu.from_numpy(state)
    if (params['net_type'] == 'CNN'):
        state = state.unsqueeze(dim=0)
    elif (params['net_type'] == 'FC'):
        state = state.flatten()
    return state