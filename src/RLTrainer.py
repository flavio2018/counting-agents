import time
from collections import OrderedDict
import pickle
import numpy as np
import torch
from MultiAgentEnvironment import MultiAgentEnvironment
from SingleAgent import SingleRLAgent
from tensorboardX import SummaryWriter
import timeit
import pytorch_utils as ptu
from datetime import timedelta
#from ExperimentSetups import *
from DQN_Agent_Single import DQN_Agent_Single
from PPO_Agent_Single import PPO_Agent_Single
from DQN_Agent_Double import DQN_Agent_Double
from ReplayMemory import ReplayMemory
from PIL import Image, ImageDraw
import utils
import pickle
import os


class RL_Trainer(object):

    def __init__(self, params):

        # Get params, create logger, create TF session
        self.params = params
        #self.logger = Logger(self.params['logdir'])
        #single_or_multi_agent = AgentSetupDict[self.params['Agent_Setup']].single_or_multi_agent
        single_or_multi_agent = self.params['single_or_multi_agent']

        if(single_or_multi_agent == 'single'):
            self.env = SingleRLAgent(params['agent_params'])
            self.env_class = SingleRLAgent
            if(self.params['agent_params']['RL_method'] == 'PPO'):
                self.agent = PPO_Agent_Single(self.env, self.params['agent_params'])
            elif(self.params['agent_params']['RL_method'] == 'DQN'):
                self.agent = DQN_Agent_Single(self.env, self.params['agent_params'])
        elif(single_or_multi_agent == 'multi'):
            self.env = MultiAgentEnvironment(params['agent_params'])
            self.env_class = MultiAgentEnvironment
            self.agent = DQN_Agent_Double(self.env, self.params['agent_params'])
            if (self.params['agent_params']['RL_method'] == 'PPO'):
                print("PPO not implemented for multi-agents yet!!")

        self.env.reset()

        self.mean_episode_reward = -float('nan')
        self.best_mean_episode_reward = -float('inf')

    def run_training_loop(self, num_iterations):
        print("Inside run_training_loop")
        print("Writer for Tensorboard on: ", self.params['logdir'])
        writer = SummaryWriter(self.params['logdir'])
        writingOn = "Experiment_Parameters"
        writer_text = str(self.params).replace(',', '<br/>')
        writer.add_text(writingOn, writer_text, 0)
        writer_text = str(self.agent.env.reward_dict).replace(',', '<br/>')
        writer.add_text(writingOn, writer_text, 0)
        model_path = self.params['logdir'] + '/model.pt'
        writer.add_text(writingOn, model_path, 0)
        dict_path = self.params['logdir'] + '/params.pickle'
        with open(dict_path, "wb") as f:
            pickle.dump(self.params, f)
        summed_loss = 0
        start_time = timeit.default_timer()
        log_loss_frequ = self.params['log_loss_frequ']
        best_mean_reward = 0.0
        itr = 0
        Is_master_all = False
        master_episodes = {}

        while(itr < num_iterations and Is_master_all == False):
            if(itr % self.params['collect_every_n_iterations'] == 0):
                _, _ = self.run_episodes_with_agent(train_episode=itr, writer=None, collect=True, eval=False, n_episodes=self.params['collect_n_episodes_per_itr'])
            if(itr % self.params['eval_every_n_iterations'] == 0):
                 print("Evaluating on training set up to ", self.agent.env.max_objects , " ... ")
                 mean_reward_train, mean_solved_train = self.run_episodes_with_agent(train_episode=itr, writer=writer, collect=False, eval=True, n_episodes=self.params['eval_n_episodes_per_itr'])
                 print('MEAN {dataset} REWARD: {mean_reward}'.format(dataset='train', mean_reward=mean_reward_train))
                 print('MEAN {dataset} SCORE: {mean_solved}'.format(dataset='train', mean_solved=mean_solved_train))
                 print("Epsilon: ", self.agent.eps)
                 if(mean_reward_train > best_mean_reward):
                     best_model_path = self.params['logdir'] + '/best_model.pt'
                     torch.save(self.agent.policy_net.state_dict(), best_model_path)
                     best_mean_reward = mean_reward_train

            #for i in range(5):
            #    loss_i = self.agent.optimize_model()
            #    #print(loss_i)
            loss = self.agent.optimize_model()

            if(loss is not None):
                summed_loss += loss
                # LOG
            if(writer is not None and itr % log_loss_frequ == 0):
                print("Average loss: ", summed_loss / log_loss_frequ, itr)
                writingOn = 'metrics/z_avg_q_value_loss_over_past_iterations'
                writer.add_scalar(writingOn, summed_loss / log_loss_frequ, itr)
                summed_loss = 0

            if (itr % self.params['eval_every_n_iterations'] == 0):
                if(mean_solved_train>0.98):
                    master_episodes[self.agent.env.max_objects] = itr
                    mean_reward_train, mean_solved_train = self.run_episodes_with_agent(train_episode=itr,
                                                                                        writer=writer, collect=False,
                                                                                        eval=True,
                                                                                        n_episodes=self.params[
                                                                                            'eval_n_episodes_per_itr'],
                                                                                        master=True)
                    if(self.agent.env.max_objects == self.params['max_max_objects']):
                        Is_master_all = True
                        # Run evaluation runs with master=True to save learned representation when task is mastered:
                    if(self.params['curriculum_learning'] and self.agent.env.max_objects < self.params['max_max_objects']):
                        print("=========================== \n ===========================")
                        print("TRAIN FROM 1 TO ", self.agent.env.max_objects + 1)
                        print("=========================== \n ===========================")

                        self.agent.env.max_objects += 1
                        max_objects = self.agent.env.max_objects
                        self.agent.env = self.env_class(self.params['agent_params'], max_objects)
                        self.agent.memory = ReplayMemory(self.params['agent_params']['MEMORY_CAPACITY'])



            itr += 1

        final_reward, final_score = self.run_episodes_with_agent(writer=writer, collect=False, eval=True, n_episodes=self.params['eval_n_episodes_per_itr'], final=True)
        print('Average Final Reward: {:.2f}'.format(final_reward))
        print('Average Final Score: {:.2f}'.format(final_score))
        # Save model
        torch.save(self.agent.policy_net.state_dict(), model_path)
        print("Model saved in: ", model_path)
        # Save master episodes
        _dir = self.params['logdir']
        with open(_dir + '/master_episodes.pkl', 'wb') as f:
            pickle.dump(master_episodes, f, pickle.HIGHEST_PROTOCOL)

        elapsed = timeit.default_timer() - start_time
        print("Elapsed time: {}".format(timedelta(seconds=elapsed)))
        writer.close()
        print("DONE")

    def run_episodes_with_agent(self, writer=None, train_episode=0, collect=False, eval=True, n_episodes=50, final=False, master=False):
        total_rewards = []
        total_solved = []
        summed_rewards = 0
        summed_totals = 0
        env = self.agent.env
        ext_repr_imgs = {i: [] for i in range(0, self.agent.env.max_objects+1)}
        ext_repr = {i: [] for i in range(0, self.agent.env.max_objects + 1)}

        for i_episode in range(n_episodes):
            state = env.reset()
            t_sofar = -1
            done = False
            episode_rewards = []
            actions_during_episode = []

            while not done:
                t_sofar += 1
                action = self.agent.select_action(state, train_episode, collect=collect, deterministic=eval)
                if (env.params['single_or_multi_agent'] == 'multi'):
                    actions_during_episode.append(action)

                #env.action = np.zeros(env.action_dim)
                #env.action[env.all_actions_dict_inv[action]] = 1
                write_each_time_step(env, eval, writer, train_episode, i_episode, t_sofar, actions_during_episode,
                                     action, master)

                next_state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                if(self.agent.params['RL_method'] == 'PPO'):
                    # saving reward and is_terminals
                    self.agent.buffer.rewards.append(reward)
                    self.agent.buffer.is_terminals.append(done)

                # Store the transition in memory
                if(collect == True):
                    state_for_memory, next_state_for_memory, action_for_memory, reward = self.agent.convert_to_memory_compatible_format(state, next_state, action, reward)
                    self.agent.memory.push(state_for_memory, action_for_memory, next_state_for_memory, reward)

                state = next_state

                if done:
                    reward = sum(episode_rewards)
                    summed_rewards += reward
                    total_rewards.append(reward)
                    total_solved.append(info['IsSolved'])
                    summed_totals += info['IsSolved']

                #write_each_time_step(env,eval, writer, train_episode, i_episode, t_sofar, actions_during_episode, action)
            ext_repr_imgs = write_after_each_episode(env, eval, writer, train_episode, i_episode, t_sofar, actions_during_episode, ext_repr_imgs, ext_repr, master)
        mean_reward = summed_rewards/n_episodes
        mean_solved = summed_totals/n_episodes
        write_after_evaluation(env, eval, writer, train_episode, i_episode, t_sofar, ext_repr_imgs, ext_repr, master, mean_reward, mean_solved, self.params['logdir'])

        return mean_reward, mean_solved



def write_each_time_step(env,eval, writer, train_episode, i_episode, t_sofar, actions_during_episode, action, master):
    if (eval and writer is not None):
        if(train_episode % 10000 == 0 or master):
            if (i_episode < 5):
                writingOn = 'whole_episode_' + str(train_episode) + '/' + str(i_episode)
                if(master):
                    writingOn = 'master_' + writingOn
                writer.add_image(writingOn, np.asarray(env.render()).astype(np.uint8).transpose([2, 0, 1]), t_sofar)

    if (eval and i_episode == 0):  #
        if (env.params['single_or_multi_agent'] == 'single'):
            if (t_sofar == 0):
                print("num: ", env.n_objects)
                if(env.params['observation'] == 'temporal'):
                    binary_event_timesteps = [0]*env.max_episode_length
                    for e_i in env.event_timesteps:
                        binary_event_timesteps[e_i] = 1
                    print("event_timesteps: ", binary_event_timesteps)
            actions_during_episode.append(env.all_actions_dict[action])
        if (env.params['single_or_multi_agent'] == 'multi'):
            if (t_sofar == 0):
                print("num: ", [env.agents[0].n_objects, env.agents[1].n_objects])
            print("act ", t_sofar, ": ", [env.agents[0].all_actions_dict[action_i] for action_i in action])

def write_after_each_episode(env, eval, writer, train_episode, i_episode, t_sofar, actions_during_episode, ext_repr_imgs, ext_repr, master):
    if (env.params['single_or_multi_agent'] == 'single'):
        if (eval and i_episode == 0):
            print("Acts: ", actions_during_episode)
    if(train_episode % 1000 == 0 or master):
        if(eval and writer is not None):
            img_height = 100
            if (env.params['single_or_multi_agent'] == 'single'):
                agenty = env
            else:
                agenty = env.agents[0]

            ## Save 1 example representation to tensorboard
            ext_img_shape = (img_height * agenty.ext_shape[1], img_height * agenty.ext_shape[0])
            ext_repr_img = Image.fromarray(agenty.ext_repr.externalrepresentation * 255).resize(ext_img_shape, resample=0)
            ext_repr_img = utils.add_grid_lines(ext_repr_img, agenty.ext_repr.externalrepresentation)
            ext_repr_img = ext_repr_img.transpose(Image.TRANSPOSE)
            annotaty = str(agenty.n_objects)
            ext_repr_img = utils.annotate_below(ext_repr_img, annotaty).convert('RGB')
            dimmy = 1 if agenty.ext_shape[1] == 1 else 2
            if (dimmy==1):
                space_img = Image.fromarray(np.ones(agenty.ext_shape[1], dtype=np.uint8)*255).resize((img_height, img_height//4), resample=0)
                ext_repr_img = utils.concat_imgs_h([ext_repr_img, space_img], dist=0)
            else:
                space_img = Image.fromarray(np.ones(agenty.ext_shape[1], dtype=np.uint8)*255).resize((img_height//4, img_height), resample=0)
                ext_repr_img = utils.concat_imgs_h([ext_repr_img, space_img], dist=0)

            #ext_repr_img = np.asarray( ext_repr_img ).astype(np.uint8).transpose([2,0,1])
            ext_repr_imgs[agenty.n_objects].append(ext_repr_img)

            ## Accumulate representations in dict:
            ext_repr[agenty.n_objects].append(agenty.ext_repr.externalrepresentation)




            return ext_repr_imgs

def write_after_evaluation(env, eval, writer, train_episode, i_episode, t_sofar, ext_repr_imgs, ext_repr, master, mean_reward, mean_solved, dir_path):
    if (train_episode % 1000 == 0 or master):
        if (eval and writer is not None):
            #ext_repr_img = np.asarray(ext_repr_img).astype(np.uint8).transpose([2, 0, 1])
            #total_imgs = [np.expand_dims(ext_repr_imgs[i][0], axis=0) for i in range(1, env.max_objects + 1)]
            total_imgs_tensor = [np.expand_dims(np.asarray(ext_repr_imgs[i][0]).astype(np.uint8).transpose([2, 0, 1]), axis=0) for i in range(0, env.max_objects + 1)]
            total_imgs = [ext_repr_imgs[i][0] for i in range(0, env.max_objects + 1)]

            min_examples = 10000
            for i in range(0, env.max_objects + 1):
                if(len(ext_repr_imgs[i]) < min_examples):
                    min_examples = len(ext_repr_imgs[i])
            example_repr_for_all_numbers_list = []
            show_n_examples = min(min_examples, 10)
            for j in range(show_n_examples):
                j_example_repr_for_all_numbers = [ext_repr_imgs[i][j] for i in range(0, env.max_objects + 1)]
                j_example_repr_for_all_numbers = utils.concat_imgs_h(j_example_repr_for_all_numbers, dist=50)
                example_repr_for_all_numbers_list.append(j_example_repr_for_all_numbers)

            all_example_repr = utils.concat_imgs_v(example_repr_for_all_numbers_list, dist=50) #.resize(env.max_objects*100,show_n_examples*100*10)

            if(env.dimmy==1):
                total_imgs_tensor = np.concatenate(total_imgs_tensor, axis=2)
                total_imgs = utils.concat_imgs_v(total_imgs, dist=10)
            elif(env.dimmy==2):
                total_imgs_tensor = np.concatenate(total_imgs_tensor, axis=0)
                total_imgs = utils.concat_imgs_h(total_imgs, dist=10)

            writingOn = 'example_representations_/'
            writingOn += "" if master == False else "master"
            writer.add_images(writingOn, total_imgs_tensor, train_episode)

            if (master):

                if (env.params['single_or_multi_agent'] == 'single'):
                    agenty = env
                else:
                    agenty = env.agents[0]

                # Create directory
                _dir = dir_path + '/ext_representations'
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

                # Save image in same directory
                file_name = _dir + '/' + agenty.params['exp_name'] + '_master_representation.png'
                total_imgs.save(file_name)

                file_name = _dir + '/' + agenty.params['exp_name'] + '_master_representations.png'
                all_example_repr.save(file_name)

                ## Save all representations to array in same directory
                # DUMP DICT OR DATAFRAME IN _dir
                with open(_dir + '/external_representations.pkl', 'wb') as f:
                    pickle.dump(ext_repr, f, pickle.HIGHEST_PROTOCOL)



    if (eval and writer is not None):
        writingOn = 'metrics/mean_rewards'
        writer.add_scalar(writingOn, mean_reward, train_episode)
        writingOn = 'metrics/mean_solved'
        writer.add_scalar(writingOn, mean_solved, train_episode)

def demonstrate_model(model, env, collect=False, eval=True, n_objects=None):
    total_rewards = []
    summed_rewards = 0

    states = env.reset(n_objects)
    states = torch.stack([torch.unsqueeze(ptu.from_numpy(state), dim=0) for state in states]).transpose(0, 1)
    t_sofar = 0
    done = False

    actions_during_episode = []
    img_list = []
    img_list.append(env.render(display_id="model_demo"))

    while not done:
        t_sofar += 1
        action = self.agent.select_action(state, train_episode, deterministic=eval)
        actions_during_episode.append(action)
        next_state, reward, done, _ = env.step(action)

        episode_rewards.append(reward)
        # Convert numpy objects to tensors for saving in memory
        reward = torch.tensor([ptu.from_numpy(np.array(reward))])
        action = ptu.from_numpy(np.array(action)).unsqueeze(0).unsqueeze(1).type(torch.int64)
        if next_state is not None:
            next_state = ptu.from_numpy(next_state)

        # Store the transition in memory
        if (collect == True):
            self.agent.memory.push(state.unsqueeze(0), action, next_state.unsqueeze(0), reward)
        # Log example actions:
        if (writer is not None and isTest == False and i_episode < 2):
            for a_i in range(self.agent.n_actions):
                writingOn = 'example_actions_' + str(i_episode) + '/' + str(a_i)
                writer.add_scalar(writingOn, action[a_i], train_episode)

        # Move to the next state
        state = next_state

        if done:
            reward = sum(episode_rewards)
            summed_rewards += reward
            total_rewards.append(reward)

        if (eval and i_episode == 0):
            print("act ", t_sofar, ": ", env.all_actions_dict[int(action.cpu().numpy().astype(int))])
        img_list.append(env.render(display_id="model_demo"))
        time.sleep(2)
    return img_list
