"""
This file contains the implementation of the environment from the point of view of a single agent. The environment class SingleRLAgent embeds three subclasses (FingerLayer, ExternalRepresentation, OtherInteractions) which implement the dynamics of the different environment parts.
"""
import time
from PIL import Image, ImageDraw
from IPython.display import display, update_display
import numpy as np
import utils
import random
import pytorch_utils as ptu
from ExperimentSetups import *

# TODO (?): later in utils
from PIL import ImageFont
#from fonts.ttf import AmaticSC

class SingleRLAgent():
    """
    This class implements the environment as a whole.
    """
    def __init__(self, agent_params, n_objects=None):
        self.params = agent_params
        self.max_objects = self.params['max_objects']
        self.max_episode_length = calc_max_episode_length(self.max_objects, self.params['observation']) if 'max_episode_length' not in self.params else self.params['max_episode_length']
        self.experiment_specific_setup = ExperimentSetup(agent_params) #AgentSetupDict[self.params['Agent_Setup']](agent_params)
        self.IsPartOfMultiAgents = True if agent_params['single_or_multi_agent'] == 'multi' else False

        self.check_reward = True
        #print("Working with max ", self.max_objects, " objects")


        model=None
        self.max_objects = agent_params['max_objects']
        self.n_objects = random.randint(1, self.max_objects) if(n_objects is None) else n_objects
        self.obs_dim = agent_params['obs_dim']
        
        # Initialize external representation (the piece of paper the agent is writing on)
        self.ext_repr = choose_external_representation(self.experiment_specific_setup.external_repr_tool, self.obs_dim) #ExternalRepresentation(self.obs_dim)

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        self.fingerlayer = FingerLayer(self.obs_dim)

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        max_n = self.params['max_max_objects'] if self.params['curriculum_learning'] else self.params['max_objects']
        self.otherinteractions = OtherInteractions(self.experiment_specific_setup.task, max_n)

        #self.state_layers = [self.ext_repr, self.fingerlayer, self.otherinteractions]
        # Initialize action
        #all_action_dicts = [state_layer.actions for state_layer in self.state_layers]
        self.all_actions_list, self.all_actions_dict = self.merge_actions([self.ext_repr.actions, self.fingerlayer.actions, self.otherinteractions.actions])
        self.rewrite_all_action_keys()
        self.action_dim = len(self.all_actions_list)

        # Initialize neural network model: maps observation-->action
        self.model = model
        self.fps_inv = 500 #ms
        self.is_submitted_ext_repr = False
        self.submitted_ext_repr = None

        self.reset()


    #def update_state(self):
    #    self.state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation, self.ext_repr_other])
    #    return self.state

    def step(self, action):
        self.timestep += 1
        # Define how actiocn interacts with environment: e.g. with observation space and external representation
        # self.obs.step(action_on_obs[action]) # no interaction with the observation space yet
        if(action in self.fingerlayer.actions):
            self.fingerlayer.step(action)

        # For action on external representation:
        if(action in self.ext_repr.actions):
            self.ext_repr.step(action, self)

        if(action in self.otherinteractions.actions):
            if (action == 'submit'):
                self.is_submitted_ext_repr = True
                self.submitted_ext_repr = self.ext_repr.externalrepresentation
            elif (action == 'larger'):
                pass
            elif (action == 'smaller'):
                pass

        # Build action-array according to the int/string action. This is mainly for the demo mode, where actions are given
        # manually by str/int. When trained action-array is input.
        self.action = np.zeros(self.action_dim)
        self.action[self.all_actions_dict_inv[action]] = 1
        self.state = self.experiment_specific_setup.update_state(self)

        if(not self.IsPartOfMultiAgents):
            if(self.check_reward):
                reward, self.done = self.experiment_specific_setup.reward_done_function(self)
            else:
                reward, self.done = 0, False
        else:
            reward, self.done = 0, False

        self.experiment_specific_setup.env_update_function(self)

        if(self.timestep > self.max_episode_length):
            self.done = True

        if(self.done):
            self.states = None
        info = None

        return self.state, reward, self.done, info


    def render(self, display_id=None):
        img_height=200
        self.obs_img = Image.fromarray(self.obs*255).resize( (img_height,img_height), resample=0)
        self.obs_img = utils.add_grid_lines(self.obs_img, self.obs)
        self.obs_img = self.obs_img.transpose(Image.TRANSPOSE)
        self.obs_img = utils.annotate_below(self.obs_img, "Observation")

        self.action_img = Image.fromarray(self.action*255).resize( (int(img_height/4),img_height), resample=0)
        self.action_img = utils.add_grid_lines(self.action_img, np.reshape(self.action, (-1, 1)))
        self.action_img = utils.annotate_nodes(self.action_img, self.all_actions_list)
        self.action_img = utils.annotate_below(self.action_img, "Action")


        self.ext_repr_img = Image.fromarray(self.ext_repr.externalrepresentation*255).resize( (img_height,img_height), resample=0)
        self.ext_repr_img = utils.add_grid_lines(self.ext_repr_img, self.ext_repr.externalrepresentation)
        self.ext_repr_img = self.ext_repr_img.transpose(Image.TRANSPOSE)
        self.ext_repr_img = utils.annotate_below(self.ext_repr_img, "External representation")

        if hasattr(self, 'ext_repr_other'):
            self.ext_repr_other_img = Image.fromarray(self.ext_repr_other*255).resize( (img_height,img_height), resample=0)
            self.ext_repr_other_img = utils.add_grid_lines(self.ext_repr_other_img, self.ext_repr_other)
            self.ext_repr_other_img = self.ext_repr_other_img.transpose(Image.TRANSPOSE)
            self.ext_repr_other_img = utils.annotate_below(self.ext_repr_other_img, "External-Other representation")

        self.finger_img = Image.fromarray(self.fingerlayer.fingerlayer*255).resize( (img_height,img_height), resample=0)
        self.finger_img = utils.add_grid_lines(self.finger_img, self.fingerlayer.fingerlayer)
        self.finger_img = self.finger_img.transpose(Image.TRANSPOSE)
        self.finger_img = utils.annotate_below(self.finger_img, "Finger layer")
        if hasattr(self, 'ext_repr_other'):
            total_img = utils.concat_imgs_h([self.obs_img, self.finger_img, self.ext_repr_img, self.ext_repr_other_img, self.action_img], dist=10).convert('RGB')
        else:
            total_img = utils.concat_imgs_h([self.obs_img, self.finger_img, self.ext_repr_img, self.action_img], dist=10).convert('RGB')
        if(display_id is not None):
            update_display(total_img, display_id=display_id)
            #time.sleep(self.fps_inv)
        return total_img

    def reset(self, n_objects=None):
        self.n_objects = random.randint(1, self.max_objects) if(n_objects is None) else n_objects

        self.experiment_specific_setup.reset(self)

        # Initialize whole state space: concatenated observation and external representation
        self.state = self.experiment_specific_setup.update_state(self)

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        max_n = self.params['max_max_objects'] if self.params['curriculum_learning'] else self.params['max_objects']
        self.otherinteractions = OtherInteractions(self.experiment_specific_setup.task, max_n)

        self.action = np.zeros(self.action_dim)

        self.done = False
        self.timestep = 0
        self.agent_0_gave_answer_already = False
        self.max_episode_length = calc_max_episode_length(self.n_objects, self.params['observation']) if 'max_episode_length' not in self.params else self.params['max_episode_length']

        return ptu.from_numpy(self.state)

    def merge_actions(self, action_dicts):
        """This function creates the actions dict for the complete environment merging the ones related to the individual environment parts.
        """
        self.all_actions_list = []
        self.all_actions_dict = {}
        _n = 0
        for _dict in action_dicts:
            rewritten_individual_dict = {}
            for key,value in _dict.items():
                if(isinstance(value, str) and value not in self.all_actions_list):
                    self.all_actions_list.append(value)
                    self.all_actions_dict[_n] = value
                    rewritten_individual_dict[_n] = value
                    _n += 1
            _dict = rewritten_individual_dict
        #self.all_actions_dict = sorted(self.all_actions_dict.items())
        self.all_actions_list = [value for key, value in self.all_actions_dict.items()]
        return self.all_actions_list, self.all_actions_dict

    def rewrite_all_action_keys(self):
        self.all_actions_dict_inv = dict([reversed(i) for i in self.all_actions_dict.items()])
        int_to_int = {}
        for key, value in self.all_actions_dict_inv.items():
            int_to_int[value] = value
        self.all_actions_dict_inv.update(int_to_int)
        # Rewrite keys of individual action-spaces, so they do not overlap in the global action space
        self.ext_repr.actions = self.rewrite_action_keys(self.ext_repr.actions)
        self.fingerlayer.actions = self.rewrite_action_keys(self.fingerlayer.actions)

    def rewrite_action_keys(self, _dict):
        """Function used to rewrite keys of individual action-spaces, so they do not overlap in the global action space.
        """
        rewritten_dict = {}
        for key, value in _dict.items():
            if(isinstance(key, int)):
                rewritten_dict[self.all_actions_dict_inv[value]] = value
        str_to_str = {}
        for key,value in rewritten_dict.items():
            str_to_str[value] = value
        rewritten_dict.update(str_to_str)
        return rewritten_dict

class FingerLayer():
    """
    This class implements the finger movement part of the environment.
    """
    def __init__(self, dim):
        self.dim = dim
        self.fingerlayer = np.zeros((dim, dim))
        self.max_x = dim-1
        self.max_y = dim-1
        self.pos_x = 0 #random.randint(0, dim-1)
        self.pos_y = 0 #random.randint(0, dim-1)
        self.fingerlayer[self.pos_x, self.pos_y] = 1
        # This dictionary translates the total action-array to the Finger-action-strings:
        # Key will be overwritten when merged with another action-space
        self.actions = {
            0: 'left',
            1: 'right',
            2: 'up',
            3: 'down'
        }
        # revd=dict([reversed(i) for i in finger_movement.items()])
        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        str_to_str = {}
        for key, value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def step(self, move_action):
        move_action_str = self.actions[move_action]
        if(move_action_str=="right"):
            #if(self.pos_x<self.max_x):
            self.pos_x = (self.pos_x + 1) % self.dim
        elif(move_action_str=="left"):
            #if(self.pos_x > 0):
            self.pos_x = (self.pos_x - 1) % self.dim
        elif(move_action_str=="up"):
            #if(self.pos_y > 0):
            self.pos_y = (self.pos_y - 1) % self.dim
        elif(move_action_str=="down"):
            #if (self.pos_y < self.max_y):
            self.pos_y = (self.pos_y + 1) % self.dim
        self.fingerlayer = np.zeros((self.dim, self.dim))
        self.fingerlayer[self.pos_x, self.pos_y] = 1


def choose_external_representation(external_representation_tool, dim):
    if(external_representation_tool == 'MoveAndWrite'):
        return MoveAndWrite(dim)
    elif(external_representation_tool == 'WriteCoord'):
        return WriteCoord(dim)
    elif(external_representation_tool == 'Abacus'):
        return Abacus(dim)
    elif(external_representation_tool == 'SpokenWords'):
        return SpokenWords(dim)
    else:
        print("No valid 'external repr. tool was given! ")

class MoveAndWrite():
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, dim):
        self.dim = dim
        self.init_externalrepresentation(dim)
        self.actions = {
            0: 'mod_point',      # Keys will be overwritten when merged with another action-space
        }
        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, dim):
        self.externalrepresentation = np.zeros((dim, dim))

    def draw(self, draw_pixels):
        self.externalrepresentation += draw_pixels

    def step(self, action, agent):
        # This line implements if ext_repr[at_curr_pos]==0 --> set it to 1. if==1 set to 0.
        if(action == 'mod_point'):
            pos_x = agent.fingerlayer.pos_x
            pos_y = agent.fingerlayer.pos_y
            self.externalrepresentation[pos_x, pos_y] = -self.externalrepresentation[pos_x, pos_y] + 1



class Abacus():
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, dim):
        self.dim = dim
        self.externalrepresentation = np.zeros((dim, dim))

        self.init_externalrepresentation(dim)

        self.actions = {
            0: 'move_token_left',      # Keys will be overwritten when merged with another action-space
            1: 'move_token_right',
        }

        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, dim):
        self.token_pos = np.zeros(dim, dtype=int)  # gives column-number of each token in each row. start out all in the left
        self.externalrepresentation = np.zeros((dim, dim))
        for rowy in range(self.dim):
            self.externalrepresentation[self.token_pos[rowy], rowy] = 1

    def step(self, action, agent):
        '''
        Move token in the row where the finger is currently positioned either to left or right
        :param action: move_token_left, move_token_right
        :param current_row: row in which finger is currently positioned
        :return:
        '''
        current_row = agent.fingerlayer.pos_y
        if(action == 'move_token_left'):
            self.token_pos[current_row] = (self.token_pos[current_row] - 1) % self.dim
        if(action == 'move_token_right'):
            self.token_pos[current_row] = (self.token_pos[current_row] + 1) % self.dim

        self.externalrepresentation = np.zeros((agent.obs_dim, agent.obs_dim))
        for rowy in range(self.dim):
            self.externalrepresentation[self.token_pos[rowy], rowy] = 1



class OtherInteractions():
    """
    This class implements the environmental responses to actions related to communication with the other agent ('submit') or to the communication of the final answer ('larger', 'smaller').
    """
    def __init__(self, task='comparison', max_n=1):
        # Define task-dependent actions. # Keys will be overwritten when merged with another action-space
        if(task == 'compare'):
            self.actions = {
                0: 'submit',
                1: 'larger',
                2: 'smaller',
             }
        elif (task == 'classify'):
            self.actions = {i: str(i) for i in range(1, max_n+1)}

        elif (task == 'produce'):
            self.actions = {
                1: '1',
            }
        else:
            print("No valid 'task' given")

        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        str_to_str = {}
        for key, value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

def calc_max_episode_length(n_objects, observation):
    if(observation == 'spatial'):
        return 2*n_objects
    elif(observation == 'temporal'):
        return 2*n_objects




# if __name__ == '__main__':
#     agent_params = {
#         'max_objects': 9,
#         'obs_dim': 4,
#     }
#
#
#     agent = SingleRLAgent(agent_params)
#     agent.render()
#     action = 'mod_point'
#     agent.step(action)
