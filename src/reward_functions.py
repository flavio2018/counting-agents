import numpy as np
import random



def reward_done_function_classify(reward_dict, agents):
    reward = 0.0

    if (agents.single_or_double == 'single'):
        if (agents.timestep <= agents.max_episode_length):
            if(not Is_agent_did_action(agents, 'wait')):
                reward += reward_dict['action_cost']
            if (agents.params['observation'] == 'temporal'):
                reward += reward_interaction_during_events(reward_dict, agents)
            if (agents.params['observation'] == 'spatial'):
                if (agents.all_actions_dict[np.where(agents.action == 1)[0][0]] in agents.ext_repr.actions or Is_agent_moved(agents)):
                    reward += reward_dict['moved_or_mod_ext'] / (agents.max_episode_length )
                if (Is_agent_said_number(agents)):
                    reward += reward_dict['said_number_before_last_time_step'] / (agents.max_episode_length )
        else:
            if Is_agent_did_action(agents, str(agents.n_objects)):
                if(not agents.params['IsSubmitButton'] or agents.IsSubmitted):
                    reward += reward_dict['main_reward']
                    agents.done = True

    if(agents.single_or_double == 'double'):
        if (agents.timestep <= agents.max_episode_length):
            if(agents.params['observation']=='temporal'):
                for agent in agents.agents:
                    reward += reward_interaction_during_events(reward_dict, agent)
            if (agents.params['observation'] == 'spatial'):
                for agent in agents.agents:
                    if (agent.all_actions_dict[np.where(agent.action == 1)[0][0]] in agent.ext_repr.actions or Is_agent_moved(agent)):
                        reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
                    if (Is_agent_said_number(agent)):
                        reward += reward_dict['said_number_before_last_time_step'] / (agent.max_episode_length )
        else:
            first_said_correct = False
            second_said_correct = False
            agent_0, agent_1 = agents.agents[0], agents.agents[1]
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
        if(agents.params['observation']=='temporal'):
            reward += reward_interaction_during_events(reward_dict, agents)
        if (agents.params['observation'] == 'spatial'):
            for agent in agents.agents:
                if (agent.all_actions_dict[np.where(agent.action == 1)[0][0]] in agent.ext_repr.actions or Is_agent_moved(agent)):
                    reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
                if (Is_agent_said_number(agent)):
                    reward += reward_dict['said_number_before_last_time_step'] / (agent.max_episode_length )
    else:
        if(agents.agents[0].n_objects == agents.agents[1].n_objects):
            if(Is_agent_did_action(agents.agents[0], 'equal') and Is_agent_did_action(agents.agents[1], 'equal')):
                reward = reward_dict['main_reward']
                agents.done = True
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
            reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
    else:
        if (agent.ext_repr.externalrepresentation.sum() == agent.obs.sum()):
            reward += reward_dict['main_reward']
            agent.done = True

    return reward, agent.done



def reward_interaction_during_events(reward_dict, agent):

        reward = 0.0
        if (agent.all_actions_dict[np.where(agent.action==1)[0][0]] in agent.ext_repr.actions):
            if((agent.timestep-1) in agent.event_timesteps):
                reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
            else:
                reward -= reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
        if ((agent.timestep-1) not in agent.event_timesteps):
            if(Is_agent_did_action(agent, 'wait')):
                reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length)
            if(Is_agent_moved(agent)):
                reward += reward_dict['moved_or_mod_ext'] / (agent.max_episode_length)
            else:
                reward -= reward_dict['moved_or_mod_ext'] / (agent.max_episode_length )
        if (Is_agent_said_number(agent)):
            reward += reward_dict['said_number_before_last_time_step'] / (agent.max_episode_length )

        return reward


RewardFunctionDict = {
    'classify': reward_done_function_classify,
    'compare': reward_done_function_comparison,
    'reproduce': reward_done_function_reproduce
}

ZeroRewardDict = {
    'moved_or_mod_ext': +0.00,
    'said_number_before_last_time_step': -0.00,
    'main_reward': +0.0,
    'gave_answer_before_answer_time': +0.0
}



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



