import numpy as np
from reward_functions import *

class ExperimentSetup():
    RewardFunctionDict = {
        'classify': reward_done_function_classify,
        'compare': reward_done_function_comparison,
        'reproduce': reward_done_function_reproduce
    }

    ObsResetFunctionDict = {
        'spatial': obs_reset_function_spatial,
        'temporal': obs_reset_function_empty
    }

    ExtResetFunctionDict = {
        'MoveAndWrite': ext_reset_function_empty,
        'Abacus': ext_reset_function_abacus,
        'WriteCoord': ext_reset_function_empty,
        'SpokenWords': None
    }

    FingerResetFunctionDict = {
        'spatial': finger_reset_function_top_left,
        'temporal': finger_reset_function_top_left
    }

    UpdateStateFunctionDict = {
        'single': update_state_function,
        'multi': update_state_function_with_other_ext_repr
    }

    EnvUpdateFunctionDict = {
        'spatial': env_update_function_nothing,
        'temporal': env_update_function_events
    }

    def __init__(self, params):
        self.single_or_multi_agent = params['single_or_multi_agent']
        self.task = params['task']
        self.observation = params['observation']
        self.external_repr_tool = params['external_repr_tool']

        self.reward_function = self.RewardFunctionDict[params['task']]
        self.obs_reset_function = self.ObsResetFunctionDict[params['observation']]
        self.ext_reset_function = self.ExtResetFunctionDict[params['external_repr_tool']]
        self.finger_reset_function = self.FingerResetFunctionDict[params['observation']]

        self.update_state_function = self.UpdateStateFunctionDict[params['single_or_multi_agent']]
        self.env_update_function = self.EnvUpdateFunctionDict[params['observation']]

        self.reward_dict = {}
        if('main_reward' in params):
            self.reward_dict = params['reward_dict']
        else:
            self.reward_dict = {
                                'moved_or_mod_ext': +0.00,
                                'said_number_before_last_time_step': -0.00,
                                'main_reward': +0.0,
                                'gave_answer_before_answer_time': +0.0
                                }

    def reward_done_function(self, agents):
        return self.reward_function(self.reward_dict, agents)

    def reset(self, agent):
        self.obs_reset_function(agent)
        agent.ext_repr.init_externalrepresentation(agent.ext_shape)
        agent.ext_repr_other = agent.ext_repr.externalrepresentation
        self.finger_reset_function(agent)

    def env_update(self, agent):
        self.env_update_function(agent)

    def update_state(self, agent):
        agent.state = self.update_state_function(agent)
        return agent.state





#########################################
## The rest is not needed.

class SpatialComparisonMoveAndWrite():
    single_or_multi_agent = 'multi'

    def __init__(self, params):
        self.task = 'comparison'
        self.observation = 'spatial'
        self.external_repr_tool = 'MoveAndWrite'

        self.reward_dict = {
            'moved_or_mod_ext': +0.12,
            'said_number': -0.1,
            'gave_answer_before_answer_time': -0.01,
            'main_reward': +1.0
        }

    def reward_done_function(self, agents):
        reward = 0.0

        if (agents.timestep <= agents.max_episode_length):
            for agent in agents.agents:
                if (Is_agent_did_action(agent, 'mod_point') or Is_agent_moved(agent)):
                    reward += self.reward_dict['moved_or_mod_ext']
                # Punish answering before answer time
                if (Is_agent_did_action(agent, 'larger') or  Is_agent_did_action(agent, 'smaller')):
                    reward += self.reward_dict['gave_answer_before_answer_time']
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
                    reward = self.reward_dict['main_reward']
                    agents.done = True

        return reward, agents.done

    def reset(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros(agent.obs_shape)
        agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1

        # Initialize external representation (the piece of paper the agent is writing on)
        agent.ext_repr.externalrepresentation = np.zeros(agent.obs_shape)

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        agent.fingerlayer.fingerlayer = np.zeros(agent.obs_shape)
        agent.fingerlayer.fingerlayer[0, 0] = 1

    def env_update(self):
        pass

    def update_state(self, agent):
        #state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        agent.state = np.stack([agent.obs, agent.fingerlayer.fingerlayer, agent.ext_repr.externalrepresentation, agent.ext_repr_other])
        return agent.state









class SpatialClassifyMoveAndWrite():
    single_or_multi_agent = 'multi'

    def __init__(self, params):
        self.task = 'classify'
        self.observation = 'spatial'
        self.external_repr_tool = 'MoveAndWrite'

        self.reward_dict = {
            'moved_or_mod_ext': +0.005,
            'said_number': -0.1,
            'gave_answer_before_answer_time': -0.005,
            'main_reward': +1.0
        }

    def reward_done_function(self, agents):
        reward = 0.0

        if (agents.timestep <= agents.max_episode_length):
            for agent in agents.agents:
                if (Is_agent_did_action(agent, 'mod_point') or Is_agent_moved(agent)):
                    reward += self.reward_dict['moved_or_mod_ext']
                if (Is_agent_said_number(agent)):
                    reward += self.reward_dict['gave_answer_before_answer_time']
        else:

            first_said_correct = False
            second_said_correct = False
            agent_0 = agents.agents[0]
            agent_1 = agents.agents[1]
            if Is_agent_did_action(agent_0, str(agent_1.n_objects)):
                first_said_correct = True
                reward += 0.5
            if Is_agent_did_action(agent_1, str(agent_0.n_objects)):
                reward += 0.5
                second_said_correct = True
            if(first_said_correct and second_said_correct):
                #reward += self.reward_dict['main_reward']
                agents.done = True


        return reward, agents.done

    def reset(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros(agent.obs_shape)
        agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1

        # Initialize external representation (the piece of paper the agent is writing on)
        agent.ext_repr.externalrepresentation = np.zeros(agent.obs_shape)
        # Empty external representation of other agent
        agent.ext_repr_other = np.zeros(agent.ext_shape)

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        agent.fingerlayer.fingerlayer = np.zeros(agent.obs_shape)
        agent.fingerlayer.fingerlayer[0, 0] = 1

    def env_update(self):
        pass

    def update_state(self, agent):
        #state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        agent.state = np.stack([agent.obs, agent.fingerlayer.fingerlayer, agent.ext_repr.externalrepresentation, agent.ext_repr_other])
        return agent.state


class SpatialReproduceMoveAndWrite_multi():
    single_or_multi_agent = 'multi'

    def __init__(self, params):
        self.task = 'reproduce'
        self.observation = 'spatial'
        self.external_repr_tool = 'MoveAndWrite'

        self.reward_dict = {
            'moved_or_mod_ext': +0.12,
            'said_number': -0.1,
            'gave_answer_before_answer_time': -0.01,
            'main_reward': +1.0
        }

    def reward_done_function(self, agents):
        reward = 0.0

        if (agent.timestep <= agent.max_episode_length):
            for agent in agents.agents:
                if (Is_agent_did_action(agent, 'mod_point') or Is_agent_moved(agent)):
                    reward += self.reward_dict['moved_or_mod_ext']
                # Punish answering before answer time
                if (Is_agent_did_action(agent, 'larger') or Is_agent_did_action(agent, 'smaller')):
                    reward += self.reward_dict['gave_answer_before_answer_time']
        else:

            # Reproduce numerosity
            first_said_correct = False
            second_said_correct = False
            agent_0 = agents.agents[0]
            agent_1 = agents.agents[1]
            if (agent_0.ext_repr.externalrepresentation.sum() == agent_0.obs.sum()):
                first_said_correct = True
                reward += 0.5
            if (agent_1.ext_repr.externalrepresentation.sum() == agent_1.obs.sum()):
                reward += 0.5
                second_said_correct = True
            if (first_said_correct and second_said_correct):
                self.done = True


            first_said_correct = False
            second_said_correct = False
            agent_0 = agents.agents[0]
            agent_1 = agents.agents[1]
            if Is_agent_did_action(agent_0, str(agent_1.n_objects)):
                first_said_correct = True
                reward += 0.5
            if Is_agent_did_action(agent_1, str(agent_0.n_objects)):
                reward += 0.5
                second_said_correct = True
            if (first_said_correct and second_said_correct):
                agents.done = True

        return reward, agents.done

    def reset(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros((agent.obs_dim, agent.obs_dim))
        agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1

        # Initialize external representation (the piece of paper the agent is writing on)
        agent.ext_repr.externalrepresentation = np.zeros((agent.obs_dim, agent.obs_dim))

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        agent.fingerlayer.fingerlayer = np.zeros((agent.obs_dim, agent.obs_dim))
        agent.fingerlayer.fingerlayer[0, 0] = 1

    def env_update(self):
        pass

    def update_state(self, agent):
        #state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        agent.state = np.stack([agent.obs, agent.fingerlayer.fingerlayer, agent.ext_repr.externalrepresentation, agent.ext_repr_other])
        return agent.state


class SpatialReproduceMoveAndWrite():
    single_or_multi_agent = 'single'
    def __init__(self, params):
        self.task = 'reproduce'
        self.observation = 'spatial'
        self.external_repr_tool = 'MoveAndWrite'

        self.reward_dict = {
            'moved_or_mod_ext': +0.005,  # working 0.01 for comparison
            'said_number': -0.1,
            'main_reward': +1.0
        }

    def reward_done_function(self, agent):
        reward = 0.0
        if (agent.timestep <= agent.max_episode_length):
            if (Is_agent_did_action(agent, 'mod_point') or Is_agent_moved(agent)):
                reward += self.reward_dict['moved_or_mod_ext']
        else:
            if (agent.ext_repr.externalrepresentation.sum() == agent.obs.sum()):
                reward += self.reward_dict['main_reward']
                agent.done = True

        return reward, agent.done

    def reset(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros((agent.obs_dim, agent.obs_dim))
        agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1

        # Initialize external representation (the piece of paper the agent is writing on)
        agent.ext_repr.externalrepresentation = np.zeros((agent.obs_dim, agent.obs_dim))

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        agent.fingerlayer.fingerlayer = np.zeros((agent.obs_dim, agent.obs_dim))
        agent.fingerlayer.fingerlayer[0, 0] = 1

    def env_update(self):
        pass

    def update_state(self):
        state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        return state














AgentSetupDict = {
    'SpatialClassifyMoveAndWrite': SpatialClassifyMoveAndWrite,
    'SpatialComparisonMoveAndWrite': SpatialComparisonMoveAndWrite,
    'SpatialReproduceMoveAndWrite': SpatialReproduceMoveAndWrite,
    'SpatialReproduceMoveAndWrite_multi': SpatialReproduceMoveAndWrite_multi
}


#####
# OLD REWARD FUNCTIONS TO TEST Q-LEARNING ALGORITHM

# Give simple reward to test q-learning algorithm
# Agent-0 has to always say 'larger'
# self.agent_0_says_larger = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']])
# if(self.agent_0_gave_answer_already):
#    if(self.agent_0_says_larger):
#            reward = 1.0
#            self.done = True
# Both agents have to always say 'larger'
# self.agent_0_says_larger = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']])
# self.agent_1_says_larger = bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']])
# if(self.agent_0_gave_answer_already and self.agent_0_gave_answer_already):
#    if(self.agent_0_says_larger and self.agent_1_says_larger):
#            reward = 1.0
#            self.done = True


# if(not self.agent_0_gave_answer_already):
#    if(bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']]) or bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['smaller']])):
#        self.agent_0_says_larger = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']])
#        self.agent_0_says_smaller = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['smaller']])
#        self.agent_0_gave_answer_already = True
# if(not self.agent_1_gave_answer_already):
#    if(bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']]) or bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['smaller']])):
#        self.agent_1_says_larger = bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']])
#        self.agent_1_says_smaller = bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['smaller']])
#        self.agent_1_gave_answer_already = True



##
# Comparison task
# if (not self.agent_0_gave_answer_already):
#     if (bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']]) or bool(
#             self.agents[0].action[self.agents[0].all_actions_dict_inv['smaller']])):
#         self.agent_0_says_larger = bool(
#             self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']])
#         self.agent_0_says_smaller = bool(
#             self.agents[0].action[self.agents[0].all_actions_dict_inv['smaller']])
#         self.agent_0_gave_answer_already = True
# if (not self.agent_1_gave_answer_already):
#     if (bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']]) or bool(
#             self.agents[1].action[self.agents[1].all_actions_dict_inv['smaller']])):
#         self.agent_1_says_larger = bool(
#             self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']])
#         self.agent_1_says_smaller = bool(
#             self.agents[1].action[self.agents[1].all_actions_dict_inv['smaller']])
#         self.agent_1_gave_answer_already = True
# if (self.agent_0_gave_answer_already and self.agent_1_gave_answer_already):
#     if (self.agent_0_says_larger is first_larger and self.agent_0_says_smaller is not first_larger):
#         if (self.agent_1_says_larger is not first_larger and self.agent_1_says_smaller is first_larger):
#             reward += 1.0
#             self.done = True