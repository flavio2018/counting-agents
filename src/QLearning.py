"""
This file contains the implementation of the optimization procedure used to train the agents' networks in a reinforcement learning setting using the Q-Learning algorithm. The code assumes the use of an external Replay Memory (implemented in ReplayMemory.py) and the use of a Policy Network and a Target Network to improve stability.

This file also contains two functions implementing the softmax and the epsilong-greedy action selection policies, used to choose the next action of the agent based on the Q Values computed by the network.

References: 
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
import torch
from torch import nn
from collections import namedtuple
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

def softmax_action_selection(state, policy_net, temperature):
    """
    Args:
        - state: The current state of the environment
        - policy_net: The Policy Network
        - temperature: The value of temperature parameter used in the softmax function
    """
    
    if temperature < 0:
        raise Exception('The temperature value must be greater than or equal to 0 ')
        
    # If the temperature is 0, just select the best action 
    # using the eps-greedy policy with epsilon = 0
    if temperature == 0:
        return eps_greedy_action_selection(state, policy_net, 0)
    
    with torch.no_grad():
        policy_net.eval()
        q_values = policy_net(state)
    
    # Apply softmax with temp
    # set a minimum to the temperature for numerical stability
    temperature = max(temperature, 1e-8) 
    softmax_out = nn.functional.softmax(- q_values/temperature, dim=0)
   
    # Sample the action using softmax output as mass pdf
    all_possible_actions = np.arange(0, softmax_out.shape[-1])
    # this samples a random element from "all_possible_actions" 
    # with the probability distribution p (softmax_out in this case)
    action =  np.random.choice(all_possible_actions, p=softmax_out.numpy())
    
    return action
        
def eps_greedy_action_selection(state, policy_net, eps):
    """
    Args:
        - state: The current state of the environment
        - policy_net: The Policy Network
        - eps: The value of Epsilon parameter
    """
    
    sample = random.random()
    if sample > eps:
        with torch.no_grad():
            policy_net.eval()
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        # we pick a random action uniformly
        return torch.tensor([[random.randrange(n_actions)]], dtype=torch.long)
    #, device=device) # TODO GPU

def get_qvalues(state, policy_net):
    with torch.no_grad():
        policy_net.eval()
        q_values = policy_net(state) # we start from 0
    
    return q_values

def eps_greedy_modified(state, policy_net, eps):
    action = 100 # any big number
    n_actions = 6
    
    while action > n_actions:
        sample = random.random()
        if sample > eps:
            with torch.no_grad():
                policy_net.eval()
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = policy_net(state).max(0)[1].item() - 1 # we start from 0
        else:
            action = random.randrange(n_actions)
    
    return action

def optimize_model(replay_memory, policy_net, target_net, loss_fn, optimizer, gamma=0.999, batch_size=100):
    """
    Args:
        - replay_memory: The Replay Memory used to make observations uncorrelated.
        - policy_net: The Policy Network
        - target_net: The Target Network
        - loss_fn: The loss function chosen.
        - optimizer: PyTorch implementation of the chosen optimization algorithm
        - gamma: Gamma parameter in the Q-Learning algorithm
        - batch_size: Size of the batch sampled from the Replay Memory
    """
    # skip optimization when there is not a sufficient number of samples 
    # in the replay memory
    if len(replay_memory) < batch_size:
        return
    transitions = replay_memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # (all the elements where the next state is not None)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state), dtype=torch.bool))
                                        # device=device # TODO GPU
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    policy_net.train()
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(batch_size) #, device=device) # TODO GPU
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute loss
    loss_val = loss_fn(state_action_values, expected_state_action_values.unsqueeze(1))
    
    print('hello!')
    print(loss_val.item(), state_action_values, expected_state_expected_state_action_values)
    
    # Optimize the model
    optimizer.zero_grad()
    loss_val.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    
    return loss_val
