import os
import random
import time
import sys
from RLTrainer import RL_Trainer
import cProfile
from plot_utils import *


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--single_or_multi_agent', choices=['single', 'multi'], type=str, default='single')
    parser.add_argument('--task', type=str, choices=['compare', 'classify', 'reproduce'], default='classify')
    parser.add_argument('--external_repr_tool', type=str, choices=['MoveAndWrite', 'WriteCoord', 'Abacus', 'SpokenWords'], default='WriteCoord')
    parser.add_argument('--observation', type=str, choices=['spatial', 'temporal'], default='temporal')


    parser.add_argument('--max_objects', type=int, default=1)
    #parser.add_argument('--max_episode_length', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=60000)
    # If curriculum_learning is True, max_object will increment by 1 from initial max_objects, whenever the agent reaches a mean
    # reward of 0.98. Incrementation will stop at max_max_objects.
    parser.add_argument('--curriculum_learning', type=bool, default=True)
    parser.add_argument('--max_max_objects', type=int, default=9)
    parser.add_argument('--obs_ext_shape', nargs='+', type=int, default=(4,4))
    parser.add_argument('--net_type', type=str, choices=['FC', 'CNN'], default='FC')
    parser.add_argument('--event_distance_range', nargs='+', type=int, default=(2, 3))

    parser.add_argument('--debug_mode', type=bool, default=True)
    parser.add_argument('--exp_name', type=str, default='TODO')

    parser.add_argument('--BATCH_SIZE', type=int, default=16)
    parser.add_argument('--PrioratizedReplayMemory', type=bool, default=False)
    parser.add_argument('--collect_every_n_iterations', type=int, default=1)
    parser.add_argument('--eval_every_n_iterations', type=int, default=100)
    parser.add_argument('--collect_n_episodes_per_itr', type=int, default=64)
    parser.add_argument('--eval_n_episodes_per_itr', type=int, default=100)
    parser.add_argument('--n_episodes_per_eval', type=int, default=10)
    parser.add_argument('--log_loss_frequ', type=int, default=100)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=20)

    parser.add_argument('--IsClip', type=bool, default=True)
    parser.add_argument('--grad_clip_value', type=int, default=10)

    parser.add_argument('--action_cost', type=float, default=-0.00)
    parser.add_argument('--moved_or_mod_ext', type=float, default=0.0)
    parser.add_argument('--said_number_before_last_time_step', type=float, default=0.0)
    parser.add_argument('--main_reward', type=float, default=1.0)

    args = parser.parse_args()
    params = vars(args)

    # Spatial Classify MoveAndWrite
    reward_dict = {
        'action_cost': params['action_cost'], #added if action is not 'wait'
        'moved_or_mod_ext': params['moved_or_mod_ext'],
        'said_number_before_last_time_step': params['said_number_before_last_time_step'],
        'main_reward': params['main_reward']
    }


    agent_params = {
        'RL_method': 'PPO',
        'net_type': params['net_type'], #FC or CNN
        'max_objects': params['max_objects'],
        'obs_shape': tuple(params['obs_ext_shape']),
        'ext_shape': tuple(params['obs_ext_shape']),
        'event_distance_range': tuple(params['event_distance_range']),
        'IsSubmitButton': False,
        'fixed_max_episode_length': 3, # if IsSubmitButton there will be a fixed maximum length until the agent can submit the answer. Can do before as well.
        'BATCH_SIZE': params['BATCH_SIZE'],
        'LEARNING_RATE': 5e-4,
        'target_update_freq': 10,
        'MEMORY_CAPACITY': 200,
        'GAMMA': 0.3,
        'pretrained_model_path': None,
        # '/home/silvester/programming/rl-single-agent-numbers/counting-agents/src/../data/TODO_13-05-2021_13-10-36/model.pt', #'/home/silvester/programming/rl-single-agent-numbers/counting-agents/src/../data/TODO_12-05-2021_10-17-59/model.pt', # or None
        'Is_pretrained_model': False,
        'exp_name': params['exp_name']
    }
    # TODO_07 - 05 - 2021_17 - 03 - 15        5 objects, non-exclusive numbers: 86%
    # TODO_12-05-2021_13-41-46 same but 90 percent
    epsilon_greedy_args = {
        'EPS_START': 0.9,
        'EPS_END': 0.2,
        'EPS_END_EPISODE': 0.04,  # 0.0: reaches eps_end at 0th episode. 1.0 reaches eps_end at the end of all episodes
    }


    agent_params = {**params, **epsilon_greedy_args, **agent_params, **reward_dict}  ### argsis
    agent_params['reward_dict'] = reward_dict
    params['agent_params'] = agent_params


    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data')

    if not (os.path.exists(data_path)):
        os.makedirs(data_path)

    #logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")

    number_string = str(params['max_objects'])
    if(params['curriculum_learning']):
        number_string += '_to_' + str(params['max_max_objects'])
    exp_name = [params['task'], params['external_repr_tool'], params['observation'], number_string, time.strftime("%d-%m-%Y_%H-%M-%S"), str(random.randint(0, 10000))]
    separator = '_'
    print(separator.join(exp_name))
    logdir = separator.join(exp_name)
    if(args.exp_name != 'TODO'):
        data_path = data_path + '/' + args.exp_name

    if not(os.path.exists(data_path)):
        os.makedirs(data_path)

    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    rl_trainer = RL_Trainer(params)
    rl_trainer.run_training_loop(params['num_iterations'])

    plot_and_save_analysis(logdir, params['exp_name'])


def plot_and_save_analysis(exp_dir, exp_name):
    # Load master episodes
    file_dir = exp_dir + '/master_episodes.pkl'
    with open(file_dir, 'rb') as f:
        master_episodes = pickle.load(f)

    # Plot and save master episodes
    file_path = exp_dir + '/' + exp_name + '_master_episodes.svg'
    plot_and_save_master_episodes(master_episodes, file_path)

    # Load external representations
    file_dir = exp_dir + '/ext_representations/external_representations.pkl'
    with open(file_dir, 'rb') as f:
        ext_repr = pickle.load(f)

    # Plot and save token positions of external representations if task == Abacus
    if('abacus_1D' in exp_name):
        file_path = exp_dir + '/' + exp_name + '_token_positions.svg'
        plot_and_save_token_positions_of_reprs(ext_repr, file_path=file_path)

    # Plot and save correlations between the external representations
    file_path = exp_dir + '/' + exp_name + '_ext_repr_correlation.svg'
    plot_and_save_external_repr_correlation(ext_repr, task_description='', file_path=file_path)



if __name__ == "__main__":
    #cProfile.run('main()', sort='tottimec')
     main()
