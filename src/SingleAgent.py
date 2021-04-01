import time
from PIL import Image
from IPython.display import display, update_display
import numpy as np
import utils

class SingleRLAgent():
    def __init__(self, model, agent_params):
        self.max_objects = agent_params['max_objects']
        self.obs_dim = agent_params['obs_dim']
        self.action_dim = agent_params['action_dim']
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        self.obs = np.zeros((self.obs_dim, self.obs_dim))
        self.obs.ravel()[np.random.choice(obs.size, self.max_objects, replace=False)] = 1
        # Initialize external representation (the piece of paper the agent is writing on)
        self.ext_repr = ExternalRepresentation(self.obs_dim)
        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        self.finger_layer = FingerLayer(self.obs_dim)
        # Initialize whole state space: concatenated observation and external representation
        self.state = np.stack([self.obs, self.finger_layer, self.ext_repr])
        # Initialize neural network model: maps observation-->action
        self.model = model
        self.fps_inv = 500 #ms
        self.submit_external_repr = False

    def step(self, action):
        # Define how actiocn interacts with environment: e.g. with observation space and external representation
        self.obs.step(action_on_obs[action])
        self.finger_layer.step(finger_movement[action])
        # For action on external representation:
        # Give as argument: either pixel-positions (1D or 2D) to draw on.
        #                   or draw/not-draw and current finger-position on which binary should be reversed
        #self.external_repr.step(action_on_ext_repr[action])
        self.submit_external_repr = action[4]  # e.g. if the 4th node represents submitting repres.

    def select_action(self):
        # Interface with Flavio's pytorch-agent:
        # output = convlstm_model(self.state)
        # action = set to discrete actions of output
        pass

    def render(self, display_id=None):
        self.obs_img = Image.fromarray(self.obs)
        self.action_img = Image.fromarray(self.action)
        self.ext_repr_img = Image.fromarray(self.ext_repr.externalrepresentation)
        self.finger_img = Image.fromarray(self.finger_layer.fingerlayer)
        total_img = utils.concat_imgs_h([self.obs_img, self.finger_img, self.ext_repr_img, self.action_img])
        if(display_id is not None):
            update_display(total_img, display_id=display_id) #.resize( (400,400), resample=0).convert('RGB')
            time.sleep(self.fps_inv)
        return total_img


    def reset(self):
        self.obs = np.zeros((self.dim, self.dim))
        self.obs.ravel()[np.random.choice(obs.size, self.max_objects, replace=False)] = 1
        self.ext_repr = np.zeros((self.dim, self.dim))



class FingerLayer():
    def __init__(self, dim):
        self.dim = dim
        self.fingerlayer = np.zeros((dim, dim))
        self.pos_x = random.randint(1, dim)
        self.pos_y = random.randint(1, dim)
        self.finger_layer[pos_x, pos_y] = 1

    def move(self, move_action):
        if(move_action=="right"):
            self.pos_x += 1
        elif(move_action=="left"):
            self.pos_x -= 1
        elif(move_action=="up"):
            self.pos_y += 1
        else(move_action=="down"):
            self.pos_y -= 1
        self.finger_layer = np.zeros((self.dim, self.dim))
        self.finger_layer[pos_x, pos_y] = 1


class ExternalRepresentation():
    def __init__(self, dim):
        self.dim = dim
        self.externalrepresentation = np.zeros((dim, dim))

    def draw(self, draw_pixels):
        self.externalrepresentation += draw_pixels
