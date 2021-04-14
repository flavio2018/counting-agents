"""
This file contains the implementation of the environment from the point of view of a single agent. The environment class SingleRLAgent embeds three subclasses (FingerLayer, ExternalRepresentation, OtherInteractions) which implement the dynamics of the different environment parts.
"""
import time
from PIL import Image, ImageDraw
from IPython.display import display, update_display
import numpy as np
import utils
import random
import torch

# TODO (?): later in utils
from PIL import ImageFont
#from fonts.ttf import AmaticSC

class SingleAgentEnv():
    """
    This class implements the environment as a whole.
    """
    def __init__(self, agent_params):
        model=None
        self.max_objects = agent_params['max_objects']
        self.obs_dim = agent_params['obs_dim']
        self.actions_dict = { n : '' for n in range(agent_params['n_actions']) } 
        
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        self.obs = np.zeros((self.obs_dim, self.obs_dim))
        self.obs.ravel()[np.random.choice(self.obs.size, self.max_objects, replace=False)] = 1
        # associated label
        num_objects = self.obs.sum(dtype=int)
        self.obs_label = np.zeros(self.max_objects)
        self.obs_label[num_objects-1] = 1
        
        # Initialize external representation (the piece of paper the agent is writing on)
        self.ext_repr = ExternalRepresentation(self.obs_dim, len(self.actions_dict), self.actions_dict)
        
        # Initialize Finger layers: Single 1 in 0-grid of shape dim x dim
        self.fingerlayer_scene = FingerLayer(self.obs_dim, len(self.actions_dict), self.actions_dict)
        self.fingerlayer_repr = FingerLayer(self.obs_dim, len(self.actions_dict), self.actions_dict)
        
        # Initialize whole state space: concatenated observation and external representation
        self.build_state()
        
        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        self.otherinteractions = OtherInteractions(len(self.actions_dict), self.actions_dict)
        
        # Initialize action
        #self.all_actions_list, self.all_actions_dict = self.merge_actions([self.ext_repr.actions, self.fingerlayer_scene.actions, self.fingerlayer_repr.actions, self.otherinteractions.actions])
        #self.all_actions_dict_inv = dict([reversed(i) for i in self.all_actions_dict.items()])
        #int_to_int = {}
        #for key, value in self.all_actions_dict_inv.items():
            #int_to_int[value] = value
        #self.all_actions_dict_inv.update(int_to_int)
        
        ## Rewrite keys of individual action-spaces, so they do not overlap in the global action space
        #self.ext_repr.actions = self.rewrite_action_keys(self.ext_repr.actions)
        #self.fingerlayer_scene.actions = self.rewrite_action_keys(self.fingerlayer_scene.actions)
        #self.fingerlayer_repr.actions = self.rewrite_action_keys(self.fingerlayer_repr.actions)

        #self.action_dim = len(self.all_actions_list)
        self.action_vec = np.zeros((len(self.actions_dict), 1))

        # Initialize neural network model: maps observation-->action
        self.model = model
        self.fps_inv = 500 #ms
        self.is_submitted_ext_repr = False
        self.submitted_ext_repr = None

    def step(self, q_values):
        # Define how action interacts with environment: e.g. with observation space and external representation
        # self.obs.step(action_on_obs[action]) # no interaction with the observation space yet
        
        done = False # signal episode ending
        reward = 0 # TODO: reward when finger on object
        
        action = self.eps_greedy_modified(q_values) #TODO: generalize

        if(action in self.fingerlayer_scene.action_codes):
            self.fingerlayer_scene.step(action, self.actions_dict)
            
        elif(action in self.fingerlayer_repr.action_codes):
            self.fingerlayer_repr.step(action, self.actions_dict)

        # For action on external representation:
        # Give as argument: either pixel-positions (1D or 2D) to draw on.
        #                   or draw_point/not-draw at the current finger-position
        elif(action in self.ext_repr.action_codes):
            self.ext_repr.draw_point([self.fingerlayer_repr.pos_x, self.fingerlayer_repr.pos_y])

        elif(action in self.otherinteractions.action_codes):
            self.otherinteractions.step(action, self.max_objects, self.obs_label)
            done = True
        
        reward = self.get_reward(q_values)

        # Build action-array according to the int/string action. This is mainly for the demo mode, where actions are given
        # manually by str/int. When trained action-array is input.
        self.action_vec = np.zeros((len(self.actions_dict), 1))
        self.action_vec[action] = 1
        
        self.build_state()
        
        return torch.Tensor(self.state), reward, done, 'info'
    
    #def select_action(self, q_values):
        ## Interface with Flavio's pytorch-agent:
        ## output = convlstm_model(self.state)
        ## action = set to discrete actions of output
        #action_space = q_values[:-num_objects]
        
    def eps_greedy_modified(self, q_values):
        action = 100 # any big number
        n_actions = 6
        eps = .1 # TODO: not-hardcoded
        
        while action > n_actions:
            sample = random.random()
            if sample > eps:
                action = q_values.max(0)[1].item() - 1 # we start from 0
            else:
                action = random.randrange(n_actions)
        
        return action
        
    def render(self, display_id=None):
        img_height=200
        self.obs_img = Image.fromarray(self.obs*255).resize( (img_height,img_height), resample=0)
        self.obs_img = utils.add_grid_lines(self.obs_img, self.obs)
        self.obs_img = self.obs_img.transpose(Image.TRANSPOSE)
        self.obs_img = utils.annotate_below(self.obs_img, "Observation")

        self.action_img = Image.fromarray(self.action_vec*255).resize( (int(img_height/4),img_height), resample=0)
        self.action_img = utils.add_grid_lines(self.action_img, np.reshape(self.action_vec, (-1, 1)))
        self.action_img = utils.annotate_nodes(self.action_img, list(self.actions_dict.values()))
        self.action_img = utils.annotate_below(self.action_img, "Action")


        self.ext_repr_img = Image.fromarray(self.ext_repr.externalrepresentation*255).resize( (img_height,img_height), resample=0)
        self.ext_repr_img = utils.add_grid_lines(self.ext_repr_img, self.ext_repr.externalrepresentation)
        self.ext_repr_img = self.ext_repr_img.transpose(Image.TRANSPOSE)
        self.ext_repr_img = utils.annotate_below(self.ext_repr_img, "External representation")

        self.finger_scene_img = Image.fromarray(self.fingerlayer_scene.fingerlayer*255).resize( (img_height,img_height), resample=0)
        self.finger_scene_img = utils.add_grid_lines(self.finger_scene_img, self.fingerlayer_scene.fingerlayer)
        self.finger_scene_img = self.finger_scene_img.transpose(Image.TRANSPOSE)
        self.finger_scene_img = utils.annotate_below(self.finger_scene_img, "Finger layer scene")
        
        self.finger_repr_img = Image.fromarray(self.fingerlayer_repr.fingerlayer*255).resize( (img_height,img_height), resample=0)
        self.finger_repr_img = utils.add_grid_lines(self.finger_repr_img, self.fingerlayer_repr.fingerlayer)
        self.finger_repr_img = self.finger_repr_img.transpose(Image.TRANSPOSE)
        self.finger_repr_img = utils.annotate_below(self.finger_repr_img, "Finger layer repr.")
        
        total_img = utils.concat_imgs_h([self.obs_img, self.finger_repr_img, self.finger_repr_img, self.ext_repr_img, self.action_img], dist=10).convert('RGB')
        
        if(display_id is not None):
            display(total_img, display_id=display_id)
            time.sleep(self.fps_inv)
        
        return total_img

    def reset(self):
        # generate new observation
        self.obs = np.zeros((self.obs_dim, self.obs_dim))
        self.obs.ravel()[np.random.choice(self.obs.size, self.max_objects, replace=False)] = 1
        # associated label
        num_objects = self.obs.sum(dtype=int)
        self.obs_label = np.zeros(self.max_objects)
        self.obs_label[num_objects-1] = 1 # vector form
        
        # reset external representation
        self.ext_repr = ExternalRepresentation(self.obs_dim)
        
        # reset finger layers
        self.fingerlayer_scene = FingerLayer(self.obs_dim)
        self.fingerlayer_repr = FingerLayer(self.obs_dim)
        
        # reset whole state
        self.build_state()
        
        return torch.Tensor(self.state)

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
    
    def build_state(self):
        self.state = np.stack([[self.obs,
                        self.fingerlayer_scene.fingerlayer,
                        self.fingerlayer_repr.fingerlayer, self.ext_repr.externalrepresentation]])
    
    def compare_labels(self, agent_label, true_label):
        """Encode here the label comparison dynamics.
        """
        label_dist = np.abs(np.argmax(agent_label) - np.argmax(true_label))
        
        return label_dist
    
    def get_reward(self, q_values):
        label_slice = q_values[-self.max_objects:]
        label_dist = self.compare_labels(label_slice, self.obs_label)
        if label_dist == 0:
            reward = 1
        if label_dist < 2:
            reward = .5
        if label_dist < 3:
            reward = .2
        else:
            reward = 0
        
        return reward
        

class FingerLayer():
    """
    This class implements the finger movement part of the environment.
    """
    def __init__(self, layer_dim, no_actions, env_actions_dict):
        self.layer_dim = layer_dim
        self.fingerlayer = np.zeros((layer_dim, layer_dim))
        self.max_x = layer_dim-1
        self.max_y = layer_dim-1
        self.pos_x = random.randint(0, layer_dim-1)
        self.pos_y = random.randint(0, layer_dim-1)
        self.fingerlayer[self.pos_x, self.pos_y] = 1
        
        # This dictionary translates the total action-array to the Finger-action-strings:
        # Key will be overwritten when merged with another action-space
        actions = ['left', 'right', 'up', 'down']
        self.action_codes = set()
        
        i = 0
        for k, v in env_actions_dict.items():
            if v == '' and i < len(actions):
                env_actions_dict[k] = actions[i]
                self.action_codes.add(k)
                i += 1
        
        
        # revd=dict([reversed(i) for i in finger_movement.items()])
        
        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        #str_to_str = {}
        #for key, value in self.actions.items():
            #str_to_str[value] = value
        #self.actions.update(str_to_str)

    def step(self, move_action, actions_dict):
        move_action_str = actions_dict[move_action]
        if(move_action_str=="right"):
            if(self.pos_x<self.max_x):
                self.pos_x += 1
        elif(move_action_str=="left"):
            if(self.pos_x > 0):
                self.pos_x -= 1
        elif(move_action_str=="up"):
            if(self.pos_y > 0):
                self.pos_y -= 1
        elif(move_action_str=="down"):
            if (self.pos_y < self.max_y):
                self.pos_y += 1
        self.fingerlayer = np.zeros((self.layer_dim, self.layer_dim))
        self.fingerlayer[self.pos_x, self.pos_y] = 1


class ExternalRepresentation():
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, layer_dim, no_actions, env_actions_dict):
        self.layer_dim = layer_dim
        self.externalrepresentation = np.zeros((layer_dim, layer_dim))
        
        actions = ['mod_point']
        self.action_codes = set()
        
        i = 0
        for k, v in env_actions_dict.items():
            if v == '' and i < len(actions):
                env_actions_dict[k] = actions[i]
                self.action_codes.add(k)
                i += 1
        
        #self.actions = {
            #0: 'mod_point',      # Keys will be overwritten when merged with another action-space
        #}
        #str_to_str = {}
        #for key,value in self.actions.items():
            #str_to_str[value] = value
        #self.actions.update(str_to_str)

    def draw(self, draw_pixels):
        self.externalrepresentation += draw_pixels

    def draw_point(self, pos):
        # This line implements if ext_repr[at_curr_pos]==0 --> set it to 1. if==1 set to 0.
        self.externalrepresentation[pos[0], pos[1]] = -self.externalrepresentation[pos[0], pos[1]] + 1


class OtherInteractions():
    """
    This class implements the environmental responses to actions related to the task (submit numerosity label).
    """
    def __init__(self, no_actions, env_actions_dict):
        
        actions = ['submit']
        self.action_codes = set()
        
        i = 0
        for k, v in env_actions_dict.items():
            if v == '' and i < len(actions):
                env_actions_dict[k] = actions[i]
                self.action_codes.add(k)
                i += 1
                
        #self.actions = {
            #0: 'submit'
        #}
        # Add each value as key as well. so in the end both integers (original keys) and strings (original values) can be input
        #str_to_str = {}
        #for key, value in self.actions.items():
            #str_to_str[value] = value
        #self.actions.update(str_to_str)

    def step(self, action, max_objects, true_label):
        if(action=='submit'):
            pass
            # TODO: what happens here?


if __name__ == '__main__':
    agent_params = {
        'max_objects': 9,
        'obs_dim': 4,
    }


    agent = SingleRLAgent(agent_params)
    agent.render()
    action = 'mod_point'
    agent.step(action)
