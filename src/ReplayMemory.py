"""
This file contains the implementation of an external Replay Memory
used to train the agents with the Q-Learning algorithm.

References:
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random

from collections import namedtuple 
from collections import deque # what we need for the replay memory


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
    
class ReplayMemory(object):

    def __init__(self, capacity):
        # Define a queue with maxlen "capacity"
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, next_state, reward):
        # Add the namedtuple to the queue
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        # Get all the samples if the requested batch_size is higher than 
        # the number of sample currently in the memory
        batch_size = min(batch_size, len(self)) 
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Return the number of samples currently stored in the memory
        return len(self.memory) 
