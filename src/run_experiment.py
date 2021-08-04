import os
import time
import sys
from RLTrainer import RL_Trainer
import cProfile


def main():

    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--single_or_multi_agent', choices=['single', 'multi'], type=str, default='multi')
    parser.add_argument('--task', type=str, choices=['compare', 'classify', 'reproduce'], default='classify')
    parser.add_argument('--external_repr_tool', type=str, choices=['MoveAndWrite', 'WriteCoord', 'Abacus', 'SpokenWords'], default='WriteCoord')
    parser.add_argument('--observation', type=str, choices=['spatial', 'temporal'], default='spatial')


    parser.add_argument('--max_objects', type=int, default=2)
    #parser.add_argument('--max_episode_length', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=10000)
    # If curriculum_learning is True, max_object will increment by 1 from initial max_objects, whenever the agent reaches a mean
    # reward of 0.98. Incrementation will stop at max_max_objects.
    parser.add_argument('--curriculum_learning', type=bool, default=True)
    parser.add_argument('--max_max_objects', type=int, default=9)

    parser.add_argument('--debug_mode', type=bool, default=True)
    parser.add_argument('--exp_name', type=str, default='TODO')

    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--PrioratizedReplayMemory', type=bool, default=False)
    parser.add_argument('--collect_every_n_iterations', type=int, default=1)
    parser.add_argument('--eval_every_n_iterations', type=int, default=100)
    parser.add_argument('--collect_n_episodes_per_itr', type=int, default=10)
    parser.add_argument('--eval_n_episodes_per_itr', type=int, default=100)
    parser.add_argument('--n_episodes_per_eval', type=int, default=10)
    parser.add_argument('--log_loss_frequ', type=int, default=100)

    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--eval_batch_size', type=int, default=20)

    parser.add_argument('--IsClip', type=bool, default=True)
    parser.add_argument('--grad_clip_value', type=int, default=10)

    args = parser.parse_args()
    params = vars(args)

    # SpatialComparisonMoveAndWrite - Rewards
    #reward_dict = {
    #    'moved_or_mod_ext': +0.12,
    #    'gave_answer_before_answer_time': -0.01,
    #    'main_reward': +1.0
    #}

    # Temporal Classify Abacus - Rewards
    #reward_dict = {
    #    'moved_or_mod_ext': +0.2,
    #    'said_number_before_last_time_step': -0.00,
    #    'main_reward': +0.6
    #}

    # Spatial Classify MoveAndWrite
    reward_dict = {
        'moved_or_mod_ext': +0.4 / 2,
        'said_number_before_last_time_step': -0.1,
        'main_reward': +0.6
    }

    # Temporal Compare Abacus
    #reward_dict = {
    #    'moved_or_mod_ext': +0.01,
    #    'gave_answer_before_answer_time': -0.01,
    #    'main_reward': +1.0
    #}

    obs_ext_shape = (3,1)

    agent_params = {
        'max_objects': params['max_objects'],
        'obs_shape': obs_ext_shape,
        'ext_shape': obs_ext_shape,
        'BATCH_SIZE': params['BATCH_SIZE'],
        'LEARNING_RATE': 1e-3,
        'target_update_freq': 10,
        'MEMORY_CAPACITY': 400,
        'GAMMA': 0.95,
        'pretrained_model_path': None,
        # '/home/silvester/programming/rl-single-agent-numbers/counting-agents/src/../data/TODO_13-05-2021_13-10-36/model.pt', #'/home/silvester/programming/rl-single-agent-numbers/counting-agents/src/../data/TODO_12-05-2021_10-17-59/model.pt', # or None
        'Is_pretrained_model': False
    }
    # TODO_07 - 05 - 2021_17 - 03 - 15        5 objects, non-exclusive numbers: 86%
    # TODO_12-05-2021_13-41-46 same but 90 percent
    epsilon_greedy_args = {
        'EPS_START': 0.9,
        'EPS_END': 0.1,
        'EPS_END_EPISODE': 0.2,  # 0.0: reaches eps_end at 0th episode. 1.0 reaches eps_end at the end of all episodes
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
    exp_name = [params['task'], params['external_repr_tool'], params['observation'], number_string, time.strftime("%d-%m-%Y_%H-%M-%S")]
    separator = '_'
    print(separator.join(exp_name))
    logdir = separator.join(exp_name)
    logdir = os.path.join(data_path, logdir)
    params['logdir'] = logdir
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    print("\n\n\nLOGGING TO: ", logdir, "\n\n\n")

    rl_trainer = RL_Trainer(params)
    rl_trainer.run_training_loop(params['num_iterations'])


if __name__ == "__main__":
    #cProfile.run('main()', sort='tottimec')
     main()
