"""
This file contains the training loop for the agents involved in the
communication/counting task. The loop involves two CountingAgents objects,
a Gym-like environment and includes the optimization procedure
based on Q-Learning.
"""

import time
import torch

from src.QLearning import optimize_model, get_qvalues


def training_loop(env, n_episodes, replay_memory, policy_net,
                  target_net, loss_fn, optimizer, log, visit_history,
                  eps=None, tau=None, gamma=0.999, target_update=10,
                  batch_size=128, CL_settings=None):
    """
    Args:
        - env: The Gym-like environment.
        - n_episodes: The number of episodes the agent is going to experience.
        - policy_net: The Policy Network
        - replay_memory: The Replay Memory used to make observations uncorrelated.
        - target_net: The Target Network
        - policy: Policy used to choose the action based on Q-values (either softmax or eps-greedy).
        - loss_fn: The loss function chosen.
        - optimizer: PyTorch implementation of the chosen optimization algorithm
        - log: A TensorBoard SummaryWriter object
        - eps: The epsilon parameter for the eps-greedy policy.
        - tau: The temperature parameter for the softmax policy.        
        - gamma: Gamma parameter in the Q-Learning algorithm for long-term reward
        - target_update: Number of episodes to wait before updating the target network
        - batch_size: Size of the batch sampled from the Replay Memory
    """
    if eps is None and tau is None:
        print("Both tau and epsilon are None.")
        return

    if CL_settings is None:
        n_iter = 0
        init_time = time.gmtime(time.time())
        run_timestamp = str(init_time.tm_mday)+str(init_time.tm_mon)+str(init_time.tm_hour)+str(init_time.tm_min)
    else:
        n_iter_cl_phase = 0
        n_iter = CL_settings["n_iter"]
        run_timestamp = CL_settings["run_timestamp"]
    
    for episode in range(n_episodes):
        # Initialize the environment and state
        state = env.reset()
        done = False
    
        while not done:
            n_iter += 1
            n_iter_cl_phase += 1 # unuseful comment
            
            q_values = get_qvalues(state, policy_net)
            next_state, reward, done, info = env.step(q_values, n_iter_cl_phase, visit_history)
            
            log.add_scalar(f'Reward_{run_timestamp}', reward, n_iter)
            
            reward = torch.tensor([reward])  # , device=device) TODO: CUDA
            
            if done:
                next_state = None
                
            # Store the transition in memory
            replay_memory.push(state, q_values, next_state, reward)
            
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss_val = optimize_model(replay_memory, batch_size, policy_net, target_net, loss_fn, optimizer, gamma)
            
            if loss_val is not None:
                log.add_scalar(f'Loss/train_{run_timestamp}', loss_val.item(), n_iter)
        
        # Update the target network every target_update episodes
        if episode % target_update == 0:
            print(f'E {episode} | Updating target network...')
            # Copy the weights of the policy network to the target network
            target_net.load_state_dict(policy_net.state_dict())

    CL_settings["n_iter"] = n_iter
    print("Done")
