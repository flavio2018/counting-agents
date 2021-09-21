'''
Code taken from: https://github.com/nalepae/cs285/blob/master/hw3/cs285/agents/dqn_agent.py
'''
import numpy as np
from QNets import N_Concat_CNNs
import torch
from ReplayMemory import ReplayMemory, NaivePrioritizedBuffer
import pytorch_utils as ptu
import torch.nn.functional as F



class DQN_Agent_Double(object):
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
        self.ob_dim  = self.env.states[0].shape  # Assumes that all agents have same state-dim as agent[0]
        self.ext_shape = self.env.agents[0].ext_shape
        dimmy = 1 if self.ext_shape[1] == 1 else 2
        self.dimmy = dimmy
        print("ob_dim: ", self.ob_dim)
        n_channels, screen_height, screen_width = self.ob_dim
        self.ac_dim = self.env.agents[0].action.shape
        self.n_actions = self.ac_dim[0]

        # Network initializations
        self.policy_net = N_Concat_CNNs(n_channels, self.n_actions, shared_policy=True, example_input = self.env.states, dim=dimmy).to(self.device)
        self.target_net = N_Concat_CNNs(n_channels, self.n_actions, shared_policy=True, example_input = self.env.states, dim=dimmy).to(self.device)

        if(agent_params['Is_pretrained_model']):
            self.policy_net.load_state_dict(torch.load(agent_params['pretrained_model_path']))
            self.target_net.load_state_dict(torch.load(agent_params['pretrained_model_path']))

        #self.policy_net = FC(n_channels, self.n_actions, shared_policy=False).to(self.device)
        #self.target_net = FC(n_channels, self.n_actions, shared_policy=False).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
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


    def select_action(self, states, i_episode, deterministic=False):
        self.eps_lin = self.eps_start - i_episode * (self.eps_start - self.eps_min)/(self.eps_end_episode*self.n_episodes)
        self.eps = max(self.eps_min, self.eps_lin)

        perform_random_action = np.random.random() < self.eps
        if(perform_random_action and not deterministic):
            actions = [np.random.randint(self.n_actions), np.random.randint(self.n_actions)]
        else:
            states_unsqueezed = torch.stack([torch.unsqueeze(ptu.from_numpy(state), dim=0) for state in states]).transpose(0, 1)
            actions_tensors = self.policy_net(states_unsqueezed)
            action_arr = [ptu.to_numpy(actions_tensor) for actions_tensor in actions_tensors]
            actions = [np.argmax(action_arr[0], axis=1)[0], np.argmax(action_arr[1], axis=1)[0]]

        return actions


    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]


    def optimize_model(self):
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
        #non_final_mask = torch.stack([non_final_mask, non_final_mask], dim=1).reshape(2*self.params['BATCH_SIZE'])
        #next_state_tensor = torch.cat(batch.next_state)
        if(len([s for s in batch.next_state if s is not None]) == 0):
            return None
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        reward_batch = torch.stack(batch.reward)
        #weights_batch = torch.stack(batch.weights)
        #indices_batch = torch.stack(batch.indices)
        state_action_values = self.policy_net(state_batch)
        state_action_values_tensor = torch.stack(state_action_values).transpose(0,1)
        action_tensor = torch.stack(batch.action).type(torch.int64)
        # CHANGED
        state_values = state_action_values_tensor.gather(2, action_tensor).mean(1).squeeze(1)

        next_state_values = torch.zeros(self.params['BATCH_SIZE'], device=self.device)
        q_sp1 = self.target_net(non_final_next_states)
        q_sp1_tensor = torch.stack(q_sp1).transpose(0,1)
        q_sp1_tensor_max = q_sp1_tensor.max(2)[0].detach()
        q_sp1_tensor_max_mean = q_sp1_tensor_max.mean(1)
        next_state_values[non_final_mask] = q_sp1_tensor_max_mean
        #next_state_values = q_sp1_tensor_max_mean                     ### !!!! without done-masking!!!!
        expected_state_action_values = (next_state_values * self.params['GAMMA']) + reward_batch.squeeze(1)

        loss = F.mse_loss(state_values, expected_state_action_values)
        #batch_weights = torch.from_numpy(weights)
        #loss_i = batch_weights * (state_values.squeeze(1) - expected_state_action_values) ** 2
        #if self.params['PrioratizedReplayMemory']:
        #    batch_indices = torch.from_numpy(indices)
        #    prios = loss_i + 1e-5
        #    self.memory.update_priorities(batch_indices, prios.data.cpu().numpy())
        # Optimize the model
        #loss = torch.sum(loss_i)
        self.optimizer.zero_grad()
        loss.backward()
        #for param in self.policy_net.parameters():
        #    if(param.grad is not None):
        #        param.grad.data.clamp_(-1, 1)
        #        bias = param
        #        bias_grad = param.grad
        #        #print("Param: ", param, " grad: ", param.grad)
        clip = 5
        if(self.Isclip):
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.clip)
        self.optimizer.step()
        if self.num_param_updates % self.target_update_freq == 0:
            #for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            #    target_param.data.copy_(param.data)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        #self.target_net = self.policy_net
        self.num_param_updates += 1


        return loss

    def convert_to_memory_compatible_format(self, states, next_states, actions, reward):
        if next_states is not None:
            next_state_for_memory = torch.stack([torch.unsqueeze(ptu.from_numpy(next_state), dim=0) for next_state in next_states]).transpose(0, 1)
        else:
            next_state_for_memory = next_states
        state_for_memory = torch.stack([torch.unsqueeze(ptu.from_numpy(state), dim=0) for state in states]).transpose(0, 1)
        #state_for_memory = state.unsqueeze(0)
        action_for_memory = torch.stack([torch.unsqueeze(ptu.from_numpy(np.array(action)),dim=0) for action in actions])
        reward = torch.tensor([ptu.from_numpy(np.array(reward))])

        return state_for_memory, next_state_for_memory, action_for_memory, reward


    def init_device(self):
        if torch.cuda.is_available() and self.agent_params['use_gpu']:
            self.device = torch.device("cuda:" + str(self.agent_params['gpu_id']))
            print("Using GPU id {}".format(self.agent_params['gpu_id']))
        else:
            self.device = torch.device("cpu")
            print("GPU not detected. Defaulting to CPU.")