'''
Code taken from: https://github.com/nalepae/cs285/blob/master/hw3/cs285/agents/dqn_agent.py
'''
import numpy as np
from QNets import N_Concat_CNNs, CNN
import torch
from ReplayMemory import ReplayMemory, NaivePrioritizedBuffer
import pytorch_utils as ptu
import torch.nn.functional as F


class DQN_Agent_Single(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.params = agent_params
        self.init_device()

        # Training parameters
        self.batch_size = agent_params['BATCH_SIZE']
        LEARNING_RATE = self.params['LEARNING_RATE']  # 1e-4 # argsis
        self.target_update_freq = self.params['target_update_freq']

        # Observation and action sizes for individual agents
        self.ob_dim  = self.env.state.shape  # Assumes that all agents have same state-dim as agent[0]
        self.ext_shape = self.env.ext_shape
        dimmy = 1 if self.ext_shape[1] == 1 else 2
        self.dimmy = dimmy

        print("ob_dim: ", self.ob_dim)
        n_channels, screen_height, screen_width = self.ob_dim
        self.ac_dim = self.env.action.shape
        self.n_actions = self.ac_dim[0]


        # Network initializations
        self.policy_net = CNN(n_channels, self.n_actions, example_input=self.env.state, dim=dimmy).to(self.device)
        self.target_net = CNN(n_channels, self.n_actions, example_input=self.env.state, dim=dimmy).to(self.device)

        if(agent_params['Is_pretrained_model']):
            self.policy_net.load_state_dict(torch.load(agent_params['pretrained_model_path']))
            self.target_net.load_state_dict(torch.load(agent_params['pretrained_model_path']))

        #self.policy_net = FC(n_channels, self.n_actions, shared_policy=False).to(self.device)
        #self.target_net = FC(n_channels, self.n_actions, shared_policy=False).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        #self.logstd = torch.nn.Parameter(ptu.from_numpy((self.n_actions))).to(self.device)
        # CHANGED
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.clip = self.params['grad_clip_value']
        self.Isclip = self.params['IsClip']
        #self.optimizer = torch.optim.SGD(self.policy_net.parameters(), lr=LEARNING_RATE, momentum=0.9)

        # Memory
        if self.params['PrioratizedReplayMemory']:
            self.memory = NaivePrioritizedBuffer(self.params['MEMORY_CAPACITY'])
        else:
            self.memory = ReplayMemory(self.params['MEMORY_CAPACITY'])
        self.eps_threshold = 0

        # Eps-greedy parameters
        self.eps_start = self.params['EPS_START']
        self.eps_min = self.params['EPS_END']
        self.eps_end_episode = self.params['EPS_END_EPISODE']
        self.n_episodes = self.params['num_iterations']
        self.target_update_freq = self.params['target_update_freq']
        self.num_param_updates = 0


    def select_action(self, state, i_episode, deterministic=False):
        self.eps_lin = self.eps_start - i_episode * (self.eps_start - self.eps_min)/(self.eps_end_episode*self.n_episodes)
        self.eps = max(self.eps_min, self.eps_lin)

        perform_random_action = np.random.random() < self.eps
        if(perform_random_action and not deterministic):
            actions = np.random.randint(self.n_actions)
        else:
            #states_unsqueezed = [torch.unsqueeze(state, dim=0) for state in states]
            if (isinstance(state, (np.ndarray, np.generic))):
                state = ptu.from_numpy(state)
            actions_tensor = self.policy_net(state.unsqueeze(0))
            action_arr = ptu.to_numpy(actions_tensor)
            actions = np.argmax(action_arr[0], axis=0)

        return actions


    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]


    def optimize_model(self):
        BATCH_SIZE = self.params['BATCH_SIZE']
        GAMMA = self.params['GAMMA']
        if len(self.memory) < self.params['BATCH_SIZE']:
            return
        transitions, indices, weights = self.memory.sample(self.params['BATCH_SIZE'])
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach() #torch.clip( self.target_net(non_final_next_states).max(1)[0].detach(), -1, 1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        #criterion = nn.SmoothL1Loss()
        #loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        # batch_weights = torch.from_numpy(weights)
        # loss_i = batch_weights * (state_action_values.squeeze(1) - expected_state_action_values) ** 2
        # if self.params['PrioratizedReplayMemory']:
        #     batch_indices = torch.from_numpy(indices)
        #     prios = loss_i + 1e-5
        #     self.memory.update_priorities(batch_indices, prios.data.cpu().numpy())
        # Optimize the model
        #loss = torch.sum(loss_i)
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    if (param.grad is not None):
        #        param.grad.data.clamp_(-1, 1)

        if (self.Isclip):
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()

        if self.num_param_updates % self.target_update_freq == 0:
            #for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            #    target_param.data.copy_(param.data)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.target_net = self.policy_net
        self.num_param_updates += 1

        return loss

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