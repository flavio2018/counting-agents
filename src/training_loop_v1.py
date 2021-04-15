"""
This file contains the training loop for the agents involved in the communication/counting task. The loop involves two CountingAgents objects, a Gym-like environment and includes the optimization procedure based on Q-Learning.
"""
from QLearning import optimize_model, get_qvalues

def training_loop(env, n_episodes, replay_memory, policy_net, target_net, policy, loss_fn, optimizer, log, eps=None, tau=None, target_update=10):
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
        - target_update: Number of episodes to wait before updating the target network
    """
    if eps == tau == None:
        return
    
    done = False
    policy_param = tau if tau != None else eps
    
    n_iter = 0
    
    for episode in range(n_episodes):
        # Initialize the environment and state
        state = env.reset()
        
        while not done:
            n_iter += 1
            
            # Choose the action following the policy
            #action = policy(state, policy_net, policy_param)
            q_values = get_qvalues(state, policy_net)
            next_state, reward, done, info = env.step(q_values)
            
            if done: next_state = None
                
            # Store the transition in memory
            replay_memory.push(state, q_values, next_state, reward)
            
            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            loss_val = optimize_model(replay_memory, policy_net, target_net, loss_fn, optimizer)
            
            log.add_scalar('Loss/train', loss_val.item(), n_iter)
        
        # Update the target network every target_update episodes
        if episode % target_update == 0:
            print('Updating target network...')
            # Copy the weights of the policy network to the target network
            target_net.load_state_dict(policy_net.state_dict())
    
    # env.close() ?
    print("Done")
