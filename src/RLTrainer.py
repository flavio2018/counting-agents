import time

from collections import OrderedDict
import pickle
import numpy as np
import torch
import gym
import sys

from cs285.infrastructure.utils import *
from cs285.infrastructure.logger import Logger

from cs285.agents.dqn_agent import DQNAgent
from cs285.envs.kuka_diverse_object_gym_env import KukaDiverseObjectEnv



class RL_Trainer(object):

    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger, create TF session
        self.params = params
        self.logger = Logger(self.params['logdir'])

        #############
        ## ENV
        #############
        self.env = OwnEnv()
        self.env.reset()

        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape if img else self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        print("ob_dim: ", ob_dim)
        self.params['agent_params']['ac_dim'] = ac_dim
        self.params['agent_params']['ob_dim'] = ob_dim


        #############
        ## AGENT
        #############
        agent_class = self.params['agent_class']
        self.agent = agent_class(self.env, self.params['agent_params'])

    def run_training_loop(self, n_iter=None, collect_policy=None, eval_policy=None,
                          buffer_name=None,
                          initial_expertdata=None, relabel_with_expert=False,
                          start_relabel_with_expert=1, expert_policy=None):
        num_iterations = self.agent.params['num_iterations'] #1000
        writer = SummaryWriter(self.params['logdir'])
        summed_q_value_loss1, summed_value_loss, summed_policy_loss, summed_predicted_new_q_value = 0,0,0,0
        start_time = timeit.default_timer()

        log_loss_frequ = self.params['log_loss_frequ']
        n_test_episodes = self.params['n_episodes_per_eval']

        for itr in range(num_iterations):

            if(itr % self.params['eval_every_n_iterations'] == 0):
                print("Evaluating on training set ... ")
                mean_reward_train = self.eval_agent(writer, itr, isTest=False, n_test_episodes=self.params['n_episodes_per_eval'])
                print("Evaluating on test set ... ")
                mean_reward_test = self.eval_agent(writer, itr, isTest=True, n_test_episodes=self.params['n_episodes_per_eval'])
                print('Mean {dataset} score: {mean_reward}'.format(dataset='train', mean_reward=mean_reward_train))
                print('Mean {dataset} score: {mean_reward}'.format(dataset='test', mean_reward=mean_reward_test))

                if (writer is not None):
                    writingOn = 'rewards/avg_train_reward_over_episodes'
                    writer.add_scalar(writingOn, mean_reward_train, itr)

                    writingOn = 'rewards/avg_test_reward_over_episodes'
                    writer.add_scalar(writingOn, mean_reward_test, itr)

            q_value_loss1, value_loss, policy_loss, predicted_new_q_value = self.agent.optimize_model()

            if(q_value_loss1 is not None):
                #summed_loss += loss
                summed_q_value_loss1 += q_value_loss1
                summed_value_loss += value_loss
                summed_policy_loss +=policy_loss
                summed_predicted_new_q_value += predicted_new_q_value.mean()

                # LOG
                if(writer is not None and itr % log_loss_frequ == 0):
                    writingOn = 'losses/avg_q_value_loss_over_past_iterations'
                    writer.add_scalar(writingOn, summed_q_value_loss1 / log_loss_frequ, itr)
                    writingOn = 'losses/avg_value_loss_over_past_iterations'
                    writer.add_scalar(writingOn, summed_value_loss / log_loss_frequ, itr)
                    writingOn = 'losses/avg_policy_loss_over_past_iterations'
                    writer.add_scalar(writingOn, summed_policy_loss / log_loss_frequ, itr)
                    writingOn = 'q_values/avg_predicted_new_q_value_over_past_iterations'
                    writer.add_scalar(writingOn, summed_predicted_new_q_value / log_loss_frequ, itr)

                    summed_q_value_loss1, summed_value_loss, summed_policy_loss, summed_predicted_new_q_value = 0, 0, 0, 0

        final_reward = self.eval_agent(writer, itr, isTest=True, n_test_episodes=200)
        print('Average Score: {:.2f}'.format(final_reward))
        elapsed = timeit.default_timer() - start_time
        print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
        writer.close()
        print("DONE")


    def eval_agent(self, writer=None, train_episode=0, isTest=False, n_test_episodes=50):
        total_rewards = []
        summed_rewards = 0

        env = self.agent.env
        env.isTest = isTest
        STACK_SIZE = self.agent.params['STACK_SIZE']


        for i_episode in range(n_test_episodes):
            # Initialize the environment and state
            #state = torch.from_numpy(env.reset())
            env.reset()
            state = self.agent.get_screen()
            stacked_states = collections.deque(STACK_SIZE * [state], maxlen=STACK_SIZE)

            #stacked_states = [state]
            #episode_transition_memory = []
            t_sofar = 0
            for t in count():
                t_sofar += 1
                stacked_states_t = torch.cat(tuple(stacked_states), dim=1)
                #stacked_states_t = state
                # Select and perform an action
                action = self.agent.select_action(stacked_states_t, i_episode, t, deterministic=isTest)
                action_ext = np.append(action, 0.0)
                _, reward, done, _ = env._step_continuous(action_ext)
                #next_state, reward, done, _ = env.step(action_ext)
                reward = torch.tensor([reward], dtype=torch.float32, device=self.agent.device)

                # Observe new state
                next_state = self.agent.get_screen()
                if not done:
                    next_stacked_states = stacked_states
                    next_stacked_states.append(next_state)
                    next_stacked_states_t = torch.cat(tuple(next_stacked_states), dim=1)
                else:
                    next_stacked_states_t = None

                # Store the transition in memory
                if(self.params['on_policy'] == True and isTest == False):
                    self.agent.memory.push(stacked_states_t, torch.tensor([action], device=self.agent.device), next_stacked_states_t, reward, torch.tensor([t], dtype=torch.float32, device=self.agent.device))
                    #episode_transition_memory.append([stacked_states_t, torch.tensor([action], device=self.agent.device), next_stacked_states_t, reward, torch.tensor([t], dtype=torch.float32, device=self.agent.device)])
                # Log example actions:
                if(writer is not None and isTest == False and i_episode < 2):
                    for a_i in range(self.agent.n_actions):
                        writingOn = 'example_actions_' + str(i_episode) + '/' + str(a_i)
                        writer.add_scalar(writingOn, action[a_i], train_episode)

                # Move to the next state
                stacked_states = next_stacked_states

                if done:
                    reward = reward.cpu().numpy().item()
                    summed_rewards += reward
                    total_rewards.append(reward)

                    #if (self.params['on_policy'] == True and isTest == False):
                    #    for t_i in range(t_sofar):
                    #        self.agent.memory.push(episode_transition_memory[t_i][0],episode_transition_memory[t_i][1],episode_transition_memory[t_i][2],episode_transition_memory[t_i][3],episode_transition_memory[t_i][4])

                    break



        mean_reward = summed_rewards/n_test_episodes

        env.isTest = False

        return mean_reward

