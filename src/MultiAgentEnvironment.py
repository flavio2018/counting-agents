import utils
from SingleAgent import SingleRLAgent
import numpy as np
from IPython.display import display, update_display
import time
import matplotlib.pyplot as plt
import pytorch_utils as ptu
from ExperimentSetups import *

class MultiAgentEnvironment():
    '''
    Implements Multiagent environment that looks like Single-Agent environment from the outside:
    usual attributes of RL environments are lists,
    e.g. env.action=[agent_1_action, agent_2_action],
         env.state=[agent_1_state, agent_2_state],
    '''
    def __init__(self, params, max_objects=None):

        self.params = params
        #self.experiment_specific_setup = AgentSetupDict[self.params['Agent_Setup']](self.params)
        self.max_objects = max_objects if max_objects is not None else self.params['max_objects']
        self.max_episode_length = calc_max_episode_length(self.max_objects, self.params['observation']) if 'max_episode_length' not in self.params else self.params['max_episode_length']
        self.experiment_specific_setup = ExperimentSetup(params)

        print("Working with max ", self.max_objects, " objects")
        self.check_reward = True

        agent_1 = SingleRLAgent(self.params, n_objects=max_objects)
        agent_2 = SingleRLAgent(self.params, n_objects=max_objects)
        self.agents = [agent_1, agent_2]

        self.reset()

    def step(self, actions):
        self.timestep += 1
        reward = 0
        next_states_list = []

        # Perform actions for both actions and get next_states
        for i in range(self.n_agents):
            #if(self.agents[i].is_submitted_ext_repr == False):
            if(isinstance(actions[i],np.ndarray)):
                actions_i = actions[i].item()
            else:
                actions_i = actions[i]
            next_state = self.agents[i].step(actions_i)
            next_states_list.append(next_state)
        if(self.check_reward):
            reward, self.done = self.experiment_specific_setup.reward_done_function(self)
        else:
            reward, self.done = 0, False


        if(self.timestep==self.max_episode_length):
            self.agents[0].fingerlayer.fingerlayer = 0.5 * np.ones(self.agents[0].ext_shape)
            self.agents[1].fingerlayer.fingerlayer = 0.5 * np.ones(self.agents[0].ext_shape)

        if (self.timestep == self.max_episode_length):
            self.agents[0].ext_repr_other = self.agents[1].ext_repr.externalrepresentation
            self.agents[1].ext_repr_other = self.agents[0].ext_repr.externalrepresentation
        else:
            self.agents[0].ext_repr_other = 0.5 * np.ones(self.agents[0].ext_shape)
            self.agents[1].ext_repr_other = 0.5 * np.ones(self.agents[0].ext_shape)

        self.agents[0].experiment_specific_setup.update_state(self.agents[0])
        self.agents[1].experiment_specific_setup.update_state(self.agents[1])


        if(self.timestep > self.max_episode_length):
            self.done = True


        self.states = [agent.state for agent in self.agents]
        if(self.done):
            self.states = None
        info = None

        return self.states, reward, self.done, info


    def reset(self, n_objects = None):
        n_range = np.arange(1, self.max_objects + 1)
        if(n_objects is None):
            n_objects = np.random.choice(n_range, 2, replace=True)

        #agent_1 = SingleRLAgent(self.params, n_objects=n_objects[0])
        #agent_2 = SingleRLAgent(self.params, n_objects=n_objects[1])
        #self.agents = [agent_1, agent_2]
        self.agents[0].reset(n_objects=n_objects[0])
        self.agents[1].reset(n_objects=n_objects[1])

        self.n_agents = len(self.agents)
        self.states = [agent.state for agent in self.agents]
        self.actions = None
        self.time_to_give_an_answer = False
        self.done = False
        self.timestep = 0
        self.action_dim = self.agents[0].action_dim


        self.agent_0_gave_answer_already = False
        self.agent_1_gave_answer_already = False

        return self.states


    def render(self, display_id=None):
        # Concatentate agents images vertically.
        # Each agents image consists of horizontally concatenated images: obs, finger layer, action, external repr.
        img_list = [agent.render() for agent in self.agents]
        total_img = utils.concat_imgs_v(img_list)
        if(display_id is not None):
            update_display(total_img, display_id=display_id)
        return total_img


if __name__ == '__main__':

    agent_params = {
        'max_objects': 4,
        'obs_dim': 4,
    }

    agent = MultiAgentEnvironment(agent_params)
    agent.render()

    print("agent.agents[0].n_objects: ", agent.agents[0].n_objects)
    print("agent.agents[1].n_objects: ", agent.agents[1].n_objects)

    if (agent.agents[0].n_objects > agent.agents[1].n_objects):
        first_larger = True
        print("First larger: ", first_larger)
        answer = ['larger', 'smaller']
    else:
        first_larger = False
        print("First larger: ", first_larger)
        answer = ['smaller', 'larger']

    action_list = [['left', 'left'], ['submit', 'left'], ['left', 'submit'], answer]
    display(agent.render(), display_id='multi_game')
    #plt.show()

    for action in action_list:
        _, reward, done, _ = agent.step(action)
        img = agent.render(display_id="multi_game")
        #img.show()
        #plt.imshow(img)
        #plt.clf()  # will make the plot window empty
        #img.close()
        time.sleep(1)
        print("Reward: ", reward)
        print("Done: ", done)




def calc_max_episode_length(n_objects, observation):
    if (observation == 'spatial'):
        return 2 * (n_objects-1)
    elif (observation == 'temporal'):
        if(n_objects<=3):
            return 1*n_objects+1
        big_timestep_range_from_n = 5
        max_time_length = min(big_timestep_range_from_n - 1, n_objects) * 2 # + max(0,max_objects-big_timestep_range_from_n)*3
        if (n_objects >= big_timestep_range_from_n):
            max_time_length += (n_objects - big_timestep_range_from_n + 1) * 3
        return max_time_length