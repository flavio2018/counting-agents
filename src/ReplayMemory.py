"""
This file contains the implementation of an external Replay Memory used to train the agents with the Q-Learning algorithm.

References:
    - https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
from collections import namedtuple 
from collections import deque # what we need for the replay memeory
import numpy as np

    
class ReplayMemory(object):

    def __init__(self, capacity):
        # Define a queue with maxlen "capacity"
        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

    def push(self, state, action, next_state, reward):
        # Add the namedtuple to the queue
        self.memory.append(self.Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        # Get all the samples if the requested batch_size is higher than 
        # the number of sample currently in the memory
        batch_size = min(batch_size, len(self))
        weights = np.ones(batch_size)
        return random.sample(self.memory, batch_size), None, weights

    def __len__(self):
        # Return the number of samples currently stored in the memory
        return len(self.memory)

# adapted from:
# https://github.com/higgsfield/RL-Adventure/blob/master/4.prioritized%20dqn.ipynb
# https://slideplayer.com/slide/12858412/
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)

        self.memory = deque(maxlen=capacity)
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

    #def push(self, state, action, reward, next_state, done):
    def push(self, state, action, next_state, reward):
        #assert state.ndim == next_state.ndim
        #state = np.expand_dims(state, 0)
        #next_state = np.expand_dims(next_state, 0)

        max_prio = self.priorities.max() if self.memory else 1.0

        if len(self.memory) < self.capacity:
            self.memory.append(self.Transition(state, action, next_state, reward))
        else:
            self.memory[self.pos] = self.Transition(state, action, next_state, reward)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.prob_alpha
        probs /= probs.sum()

        batch_size = min(batch_size, len(self))
        #return random.sample(self.memory, batch_size)
        indices = np.random.choice(len(self.memory), batch_size, p=probs)
        samples = [self.memory[idx] for idx in indices]
        #batch = zip(*samples)
        batch = samples

        total = len(self.memory)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)


        #states = np.concatenate(batch[0])
        #actions = batch[1]
        #rewards = batch[2]
        #next_states = np.concatenate(batch[3])
        #dones = batch[4]

        return batch, indices, weights
        #return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)

