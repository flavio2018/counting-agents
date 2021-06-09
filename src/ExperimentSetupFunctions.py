import numpy as np
import random

def reward_done_function_classify(reward_dict, agents):
    reward = 0.0
    if (agents.timestep <= agents.max_episode_length):
        for agent in agents.agents:
            if (agent.all_actions_dict[np.where(agent.action==1)[0][0]] in agent.ext_repr.actions or Is_agent_moved(agent)):
                reward += reward_dict['moved_or_mod_ext']
            if (Is_agent_said_number(agent)):
                reward += reward_dict['said_number_before_last_time_step']
    else:
        first_said_correct = False
        second_said_correct = False
        agent_0, agent_1  = agents.agents[0], agents.agents[1]
        if Is_agent_did_action(agent_0, str(agent_1.n_objects)):
            first_said_correct = True
            reward += reward_dict['main_reward']/2.0
        if Is_agent_did_action(agent_1, str(agent_0.n_objects)):
            reward += reward_dict['main_reward']/2.0
            second_said_correct = True
        if(first_said_correct and second_said_correct):
            #reward += reward_dict['main_reward']
            agents.done = True
    return reward, agents.done




def reward_done_function_comparison(reward_dict, agents):
    reward = 0.0

    if (agents.timestep <= agents.max_episode_length):
        for agent in agents.agents:
            if (agent.all_actions_dict[np.where(agent.action==1)[0][0]] in agent.ext_repr.actions or Is_agent_moved(agent)):
                reward += reward_dict['moved_or_mod_ext']
            # Punish answering before answer time
            if (Is_agent_did_action(agent, 'larger') or  Is_agent_did_action(agent, 'smaller')):
                reward += reward_dict['gave_answer_before_answer_time']
    else:
        first_larger = False
        if (agents.agents[0].n_objects > agents.agents[1].n_objects):
            first_larger = True
        agent_0_answered_larger = Is_agent_did_action(agents.agents[0], 'larger')
        agent_0_answered_smaller = Is_agent_did_action(agents.agents[0], 'smaller')
        agent_1_answered_larger = Is_agent_did_action(agents.agents[1], 'larger')
        agent_1_answered_smaller = Is_agent_did_action(agents.agents[1], 'smaller')


        if(agent_0_answered_larger is first_larger and agent_0_answered_smaller is not first_larger):
            if(agent_1_answered_larger is not first_larger and agent_1_answered_smaller is first_larger):
                reward = reward_dict['main_reward']
                agents.done = True

    return reward, agents.done


def reward_done_function_reproduce(reward_dict, agent):
    reward = 0.0
    if (agent.timestep <= agent.max_episode_length):
        if (agent.all_actions_dict[np.where(agent.action==1)[0][0]] in agent.ext_repr.actions or Is_agent_moved(agent)):
            reward += reward_dict['moved_or_mod_ext']
    else:
        if (agent.ext_repr.externalrepresentation.sum() == agent.obs.sum()):
            reward += reward_dict['main_reward']
            agent.done = True

    return reward, agent.done

def obs_reset_function_spatial(agent):
    # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
    agent.obs = np.zeros((agent.obs_dim, agent.obs_dim))
    agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1

def obs_reset_function_empty(agent):
    # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
    agent.obs = np.zeros((agent.obs_dim, agent.obs_dim))
    agent.default_obs = agent.obs
    agent.event_timesteps = random.sample(range(1, agent.max_episode_length-1), agent.n_objects)
    agent.event_obs = np.zeros((agent.obs_dim, agent.obs_dim))
    middle = agent.obs_dim//2
    for x in range(middle - 1, middle+1):
        for y in range(middle - 1, middle+1):
            agent.event_obs[x, y] = 1

def ext_reset_function_empty(agent):
    # Initialize external representation (the piece of paper the agent is writing on)
    agent.ext_repr.externalrepresentation = np.zeros((agent.obs_dim, agent.obs_dim))
    agent.ext_repr_other = np.zeros((agent.obs_dim, agent.obs_dim))

def ext_reset_function_abacus(agent):
    # Initialize external representation (the piece of paper the agent is writing on)
    agent.ext_repr.externalrepresentation = np.zeros((agent.obs_dim, agent.obs_dim))
    agent.ext_repr_other = np.zeros((agent.obs_dim, agent.obs_dim))

def finger_reset_function_top_left(agent):
    # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
    agent.fingerlayer.fingerlayer = np.zeros((agent.obs_dim, agent.obs_dim))
    agent.fingerlayer.fingerlayer[0, 0] = 1


def update_state_function_with_other_ext_repr(agent):
    #state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
    agent.state = np.stack([agent.obs, agent.fingerlayer.fingerlayer, agent.ext_repr.externalrepresentation, agent.ext_repr_other])
    return agent.state

def update_state_function(agent):
    #state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
    agent.state = np.stack([agent.obs, agent.fingerlayer.fingerlayer, agent.ext_repr.externalrepresentation])
    return agent.state


def env_update_function_nothing(agent):
    pass

def env_update_function_events(agent):
    if(agent.timestep in agent.event_timesteps):
        agent.obs = agent.event_obs
    else:
        agent.obs = agent.default_obs




#########################
## Auxiliary Functions
########################

def Is_agent_did_action(agent, action_str):
    '''
    Check if agent did action with action-key 'action_str'. If action_str is a list the function checks if the agent did
    *any* of the listed actions.
    :param action_str:
    :return: boolean: true if did action. false else.
    '''
    if action_str is list:
        agent_did_action = False
        for action_str_i in action_str:
            if(bool(agent.action[agent.all_actions_dict_inv[action_str_i]])):
                agent_did_action = True
    else:
        agent_did_action = bool(agent.action[agent.all_actions_dict_inv[action_str]])
    return agent_did_action

def Is_agent_moved(agent):
    agent.agent_left = Is_agent_did_action(agent, 'left')
    agent.agent_right = Is_agent_did_action(agent, 'right')
    agent.agent_up = Is_agent_did_action(agent, 'up')
    agent.agent_down = Is_agent_did_action(agent, 'down')
    if(agent.agent_left or agent.agent_right or agent.agent_up or agent.agent_down):
        return True
    else:
        return False


def Is_agent_said_number(agent):
    agent.said_number = False
    for n_i in range(agent.max_objects):
        if Is_agent_did_action(agent, str(n_i + 1)):
            agent.said_number = True
    return agent.said_number
