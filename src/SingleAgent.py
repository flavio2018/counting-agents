"""
This file contains the implementation of the environment from the point of view of a single agent. The environment class SingleRLAgent embeds three subclasses (FingerLayer, ExternalRepresentation, OtherInteractions) which implement the dynamics of the different environment parts.
"""
from PIL import Image, ImageDraw
from IPython.display import display, update_display
import utils
import pytorch_utils as ptu

from reward_functions import *


class SingleRLAgent():
    """
    This class implements the environment as a whole.
    """
    def __init__(self, agent_params, n_objects=None):
        self.single_or_double = 'single'
        self.params = agent_params
        self.max_objects = n_objects if n_objects is not None else self.params['max_objects']
        #self.max_episode_length = calc_max_episode_length(self.max_objects, self.params['observation'], self) if 'max_episode_length' not in self.params else self.params['max_episode_length']
        #self.experiment_specific_setup = ExperimentSetup(agent_params) #AgentSetupDict[self.params['Agent_Setup']](agent_params)
        self.IsPartOfMultiAgents = True if agent_params['single_or_multi_agent'] == 'multi' else False

        self.check_reward = True
        model=None
        self.n_objects = random.randint(1, self.max_objects) if(n_objects is None) else n_objects
        self.obs_shape = agent_params['obs_shape']
        self.ext_shape = agent_params['ext_shape']
        self.obs_external_world = ObsExternalWorld(agent_params['observation'])

        # Initialize external representation (the piece of paper the agent is writing on)
        self.ext_repr = choose_external_representation(self.params['external_repr_tool'], self.ext_shape) #ExternalRepresentation(self.obs_dim)

        # Initialize Finger layer: Single 1 in 0-grid of shape dim x dim
        self.fingerlayer = FingerLayer(self.ext_shape)

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        max_n = self.params['max_max_objects'] if self.params['curriculum_learning'] else self.params['max_objects']
        self.otherinteractions = OtherInteractions(self, self.params['task'], max_n)

        # Initialize action
        self.all_actions_list, self.all_actions_dict = self.merge_actions([self.ext_repr.actions, self.fingerlayer.actions, self.otherinteractions.actions])
        self.rewrite_all_action_keys()
        self.action_dim = len(self.all_actions_list)

        # Initialize neural network model: maps observation-->action
        self.model = model
        self.fps_inv = 500 #ms
        self.is_submitted_ext_repr = False
        self.submitted_ext_repr = None

        self.reward_dict = self.params['reward_dict'] if 'main_reward' in self.params else ZeroRewardDict
        self.reward_done_function = RewardFunctionDict[self.params['task']]
        dimmy = 1 if self.ext_shape[1] == 1 else 2
        self.dimmy = dimmy

        self.reset(self.n_objects)


    def step(self, action):
        self.timestep += 1

        if(type(action) != str):
            action = self.all_actions_dict[action]

        if(action in self.fingerlayer.actions):
            self.fingerlayer.step(action)
        if(action in self.ext_repr.actions):
            self.ext_repr.step(action, self)
        if(action in self.otherinteractions.actions):
            self.otherinteractions.step(action, self)
            #if (action == 'submit'):
            #    self.is_submitted_ext_repr = True
            #    self.submitted_ext_repr = self.ext_repr.externalrepresentation
            # elif (action == 'larger'):
            #     pass
            # elif (action == 'smaller'):
            #     pass

        # Build action-array according to the int/string action. This is mainly for the demo mode, where actions are given
        # manually by str/int.
        self.action = np.zeros(self.action_dim)
        self.action[self.all_actions_dict_inv[action]] = 1


        if(not self.IsPartOfMultiAgents and self.check_reward):
                reward, self.done = self.reward_done_function(self.reward_dict, self)
        else:
                reward, self.done = 0, False

        self.solved = self.done

        self.env_update_function()
        self.state = self.update_state()

        if(self.timestep > self.max_episode_length):
            self.done = True

        if(self.done):
            self.states = None

        info = {'IsSolved': self.solved}

        return self.state, reward, self.done, info

    def update_state(self):
        if(self.IsPartOfMultiAgents):
            self.state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation, self.ext_repr_other])
        else:
            self.state = np.stack([self.obs, self.fingerlayer.fingerlayer, self.ext_repr.externalrepresentation])
        return self.state

    def env_update_function(self):
        if(self.params['observation'] == 'temporal'):
            if (self.timestep in self.event_timesteps):
                self.obs = self.event_obs
            else:
                self.obs = self.default_obs

        if(self.IsPartOfMultiAgents== False):
            if(self.timestep>=self.max_episode_length):
                self.fingerlayer.fingerlayer = 0.5 * np.ones(self.ext_shape)
                self.obs = 0.5 * np.ones(self.ext_shape)
                #self.ext_repr.externalrepresentation = 0.5 * np.ones(self.ext_shape)

    def render(self, display_id=None):
        img_width = 50
        img_height = 50
        obs_img_shape = (img_width * self.obs_shape[1], img_height * self.obs_shape[0])
        ext_img_shape = (img_width * self.ext_shape[1], img_height * self.ext_shape[0])


        self.obs_img = Image.fromarray(self.obs*255).resize( obs_img_shape, resample=0)
        self.obs_img = utils.add_grid_lines(self.obs_img, self.obs)
        self.obs_img = self.obs_img.transpose(Image.TRANSPOSE)
        self.obs_img = utils.annotate_below(self.obs_img, "Observation")

        self.action_img = Image.fromarray(self.action*255).resize( (int(img_height),img_height * self.obs_shape[1]), resample=0)
        self.action_img = utils.add_grid_lines(self.action_img, np.reshape(self.action, (-1, 1)))
        self.action_img = utils.annotate_nodes(self.action_img, self.all_actions_list)
        self.action_img = utils.annotate_below(self.action_img, "Action")

        self.ext_repr_img = Image.fromarray(self.ext_repr.externalrepresentation*255).resize( ext_img_shape, resample=0)
        self.ext_repr_img = utils.add_grid_lines(self.ext_repr_img, self.ext_repr.externalrepresentation)
        self.ext_repr_img = self.ext_repr_img.transpose(Image.TRANSPOSE)
        self.ext_repr_img = utils.annotate_below(self.ext_repr_img, "External representation")

        if self.IsPartOfMultiAgents:
            self.ext_repr_other_img = Image.fromarray(self.ext_repr_other*255).resize( ext_img_shape, resample=0)
            self.ext_repr_other_img = utils.add_grid_lines(self.ext_repr_other_img, self.ext_repr_other)
            self.ext_repr_other_img = self.ext_repr_other_img.transpose(Image.TRANSPOSE)
            self.ext_repr_other_img = utils.annotate_below(self.ext_repr_other_img, "External-Other representation")

        self.finger_img = Image.fromarray(self.fingerlayer.fingerlayer*255).resize( ext_img_shape, resample=0)
        self.finger_img = utils.add_grid_lines(self.finger_img, self.fingerlayer.fingerlayer)
        self.finger_img = self.finger_img.transpose(Image.TRANSPOSE)
        self.finger_img = utils.annotate_below(self.finger_img, "Finger layer")
        if self.IsPartOfMultiAgents:
            total_img = utils.concat_imgs_h([self.obs_img, self.finger_img, self.ext_repr_img, self.ext_repr_other_img, self.action_img], dist=10).convert('RGB')
        else:
            total_img = utils.concat_imgs_h([self.obs_img, self.finger_img, self.ext_repr_img, self.action_img], dist=10).convert('RGB')
        if(display_id is not None):
            update_display(total_img, display_id=display_id)
            #time.sleep(self.fps_inv)
        return total_img

    def reset(self, n_objects=None):

        #dominant_prob = 0.3
        #probs = [(1 - dominant_prob) / (self.max_objects) if i < self.max_objects else dominant_prob for i in range(self.max_objects + 1)]
        #self.n_objects = np.random.choice(np.arange(0, self.max_objects + 1), p=probs)

        self.n_objects = random.randint(0, self.max_objects) if(n_objects is None) else n_objects
        #self.max_episode_length = calc_max_episode_length(self, self.n_objects, self.params['observation'])

        self.IsSubmitted = False

        #self.experiment_specific_setup.reset(self)
        self.ext_repr.reset()
        self.ext_repr_other = 0.5 * np.ones(self.ext_shape) #self.ext_repr.externalrepresentation
        self.fingerlayer.reset()
        self.obs_external_world.reset(self)

        # Initialize whole state space: concatenated observation and external representation
        self.state = self.update_state()

        # Initialize other interactions: e.g. 'submit', 'larger'/'smaller,
        max_n = self.params['max_max_objects'] if self.params['curriculum_learning'] else self.params['max_objects']
        self.otherinteractions = OtherInteractions(self, self.params['task'], max_n)

        self.action = np.zeros(self.action_dim)

        self.done = False
        self.timestep = 0
        self.agent_0_gave_answer_already = False
        #self.max_episode_length = calc_max_episode_length(self.n_objects, self.params['observation']) if 'max_episode_length' not in self.params else self.params['max_episode_length']

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








class ObsExternalWorld():
    def __init__(self, obstype):
        if(obstype=='spatial'):
            self.reset = self.obs_reset_function_spatial
        if(obstype=='temporal'):
            self.reset = self.obs_reset_function_empty

    def obs_reset_function_spatial(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros(agent.obs_shape)
        agent.obs.ravel()[np.random.choice(agent.obs.size, agent.n_objects, replace=False)] = 1
        agent.max_episode_length = calc_max_episode_length(agent, agent.n_objects, agent.params['observation'])

    def obs_reset_function_empty(self, agent):
        # Initialize observation: 1-max_objects randomly placed 1s placed on a 0-grid of shape dim x dim
        agent.obs = np.zeros(agent.obs_shape)
        agent.default_obs = agent.obs
        agent.event_timesteps = calc_event_timesteps(agent.n_objects, event_distance_range=agent.params['event_distance_range'])  #
        #if(agent.n_objects==0):
        #    agent.max_episode_length = 4
        #else:
        #    agent.max_episode_length = agent.event_timesteps[-1] + 1
        agent.max_episode_length = calc_max_episode_length(agent, agent.n_objects, agent.params['observation'], event_distance_range=agent.params['event_distance_range'])
        agent.event_obs = np.zeros(agent.obs_shape)
        middle_x = agent.obs_shape[0] // 2
        middle_y = agent.obs_shape[1] // 2
        for x in range(middle_x - 1, middle_x + 1):
            for y in range(middle_y - 1, middle_y + 1):
                agent.event_obs[x, y] = 1


class FingerLayer():
    """
    This class implements the finger movement part of the environment.
    """
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.fingerlayer = np.zeros(ext_shape)
        self.max_x = ext_shape[0]-1
        self.max_y = ext_shape[1]-1
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
            self.pos_x = (self.pos_x + 1) % self.ext_shape[0]
        elif(move_action_str=="left"):
            #if(self.pos_x > 0):
            self.pos_x = (self.pos_x - 1) % self.ext_shape[0]
        elif(move_action_str=="up"):
            #if(self.pos_y > 0):
            self.pos_y = (self.pos_y - 1) % self.ext_shape[1]
        elif(move_action_str=="down"):
            #if (self.pos_y < self.max_y):
            self.pos_y = (self.pos_y + 1) % self.ext_shape[1]
        self.fingerlayer = np.zeros(self.ext_shape)
        self.fingerlayer[self.pos_x, self.pos_y] = 1

    def reset(self):
        self.pos_x = 0 #random.randint(0, dim-1)
        self.pos_y = 0 #random.randint(0, dim-1)
        self.fingerlayer = np.zeros(self.ext_shape)
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

# Parent Class ExternalTool. Not usefully used so far. See empty fct-declarations
class ExternalTool():
    def __init__(self):
        pass
    def init_externalrepresentation(self):
        pass
    def step(self):
        pass
    def reset(self):
        pass

class MoveAndWrite(ExternalTool):
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.init_externalrepresentation(ext_shape)
        self.actions = {
            0: 'mod_point',      # Keys will be overwritten when merged with another action-space
        }
        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.externalrepresentation = np.zeros(ext_shape)

    def draw(self, draw_pixels):
        self.externalrepresentation += draw_pixels

    def step(self, action, agent):
        # This line implements if ext_repr[at_curr_pos]==0 --> set it to 1. if==1 set to 0.
        if(action == 'mod_point'):
            pos_x = agent.fingerlayer.pos_x
            pos_y = agent.fingerlayer.pos_y
            self.externalrepresentation[pos_x, pos_y] = -self.externalrepresentation[pos_x, pos_y] + 1

    def reset(self):
        self.externalrepresentation = np.zeros(self.ext_shape)



class WriteCoord(ExternalTool):
    """
    This class implements the external representation in the environment.
    Right now this external tool is only possible for 1D.
    """
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.init_externalrepresentation(ext_shape)
        self.actions = {}
        for coord in range(ext_shape[0]):
            self.actions[coord] = "write_on_" + str(coord)
        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.externalrepresentation = np.zeros(ext_shape)

    def step(self, action, agent):
        # This line implements if ext_repr[coord]==0 --> set it to 1. if==1 set to 0.
        coord_int = int(action[-1])
        self.externalrepresentation[coord_int, 0] = 1 #If you want to be able to delete as well use: -self.externalrepresentation[coord_int, 0] + 1

    def reset(self):
        self.externalrepresentation = np.zeros(self.ext_shape)

class Abacus(ExternalTool):
    """
    This class implements the external representation in the environment.
    """
    def __init__(self, ext_shape):
        self.ext_shape = ext_shape
        self.externalrepresentation = np.zeros(ext_shape)

        self.init_externalrepresentation(ext_shape)

        self.actions = {
            0: 'move_token_left',      # Keys will be overwritten when merged with another action-space
            1: 'move_token_right',
        }

        str_to_str = {}
        for key,value in self.actions.items():
            str_to_str[value] = value
        self.actions.update(str_to_str)

    def init_externalrepresentation(self, ext_shape):
        self.token_pos = np.zeros(ext_shape, dtype=int)  # gives column-number of each token in each row. start out all in the left
        self.externalrepresentation = np.zeros(ext_shape)
        for rowy in range(self.ext_shape[1]):
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
            self.token_pos[current_row] = (self.token_pos[current_row] - 1) % self.ext_shape[0]
        if(action == 'move_token_right'):
            self.token_pos[current_row] = (self.token_pos[current_row] + 1) % self.ext_shape[0]

        for col in range(self.ext_shape[0]):
            self.externalrepresentation[col, current_row] = 0
        self.externalrepresentation[self.token_pos[current_row], current_row] = 1

    def reset(self):
        self.token_pos = np.zeros(self.ext_shape, dtype=int)  # gives column-number of each token in each row. start out all in the left
        self.externalrepresentation = np.zeros(self.ext_shape)
        for rowy in range(self.ext_shape[1]):
            self.externalrepresentation[self.token_pos[rowy], rowy] = 1

class OtherInteractions():
    """
    This class implements the environmental responses to actions related to communication with the other agent ('submit') or to the communication of the final answer ('larger', 'smaller').
    """
    def __init__(self, agent, task='comparison', max_n=1):
        # Define task-dependent actions. # Keys will be overwritten when merged with another action-space
        if(task == 'compare'):
            self.actions = {
                0: 'submit',
                1: 'larger',
                2: 'smaller',
                3: 'equal',
             }
        elif (task == 'classify'):
            self.actions = {i: str(i) for i in range(0, max_n+1)}
            self.actions[max_n + 1] = 'wait'
            if(agent.params['IsSubmitButton']):
                self.actions[max_n + 2] = 'submit'

        elif (task == 'reproduce'):
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

    def step(self, action, agent):
        if(action == 'submit'):
            if(not agent.IsSubmitted):
                agent.timestep = agent.max_episode_length-1
            agent.IsSubmitted = True


def calc_max_episode_length(agent, n_objects, observation, event_distance_range=None):
    if (observation == 'spatial'):
        '''
        if(agent.params['IsSubmitButton'] or agent.params['fixed_max_episode_length']>0):
            return agent.params['fixed_max_episode_length']
        else:
            return (n_objects)
        '''
        return agent.params['fixed_max_episode_length']
    elif (observation == 'temporal'):
        '''
        if(n_objects<=3):
            return 2*n_objects+1
        big_timestep_range_from_n = 5
        max_time_length = min(big_timestep_range_from_n - 1, n_objects) * 2 # + max(0,max_objects-big_timestep_range_from_n)*3
        if (n_objects >= big_timestep_range_from_n):
            max_time_length += (n_objects - big_timestep_range_from_n + 1) * 3
        '''
        if(event_distance_range is not None):
            time_per_object =  event_distance_range[1]
        else:
            time_per_object = 3
        return time_per_object*n_objects+time_per_object


def calc_event_timesteps(n_objects, max_episode_length=None, event_distance_range=None):
    #if(n_objects<=3):
    #    return random.sample(range(1, max_episode_length), n_objects)
    big_timestep_range_from_n = 5
    if(event_distance_range is None):
        event_distance_range = [2,3]
    small_timestep_range = event_distance_range
    big_timestep_range = event_distance_range
    timestep_range = small_timestep_range
    event_timesteps = []
    t_n = 0

    for n in range(1, n_objects + 1):
        if (n == big_timestep_range_from_n):
            timestep_range = big_timestep_range
        t_n += random.randint(timestep_range[0], timestep_range[1])
        event_timesteps.append(t_n)
    return event_timesteps