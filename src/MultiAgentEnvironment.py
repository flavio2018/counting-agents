import utils
from SingleAgent import SingleRLAgent

class MultiAgentEnvironment():
    '''
    Implements Multiagent environment that looks like Single-Agent environment from the outside:
    usual attributes of RL environments are lists,
    e.g. env.action=[agent_1_action, agent_2_action],
         env.state=[agent_1_state, agent_2_state],
    '''
    def __init__(self, params):
        agent_1 = SingleRLAgent(params)
        agent_2 = SingleRLAgent(params)
        self.agents = [agent_1, agent_2]
        self.n_agents = len(self.agents)
        self.states = [agent.state for agent in self.agents]
        self.actions = None
        self.time_to_give_an_answer = False
        self.done = False

    def step(self, actions):
        reward = 0
        next_states_list = []
        for i in range(self.n_agents):
            if(agents[i].is_submitted_ext_repr == False):
                next_state = self.agents[i].step(actions[i])
                next_states_list.append(next_state)


        if(self.time_to_give_an_answer):
            # TODO: what if same n_objects for both agents?
            first_larger = False
            if(self.agents[0].n_objects > self.agents[0].n_objects):
                first_larger = True
            agent_0_says_larger = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['larger']])
            agent_1_says_larger = bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['larger']])
            agent_0_says_smaller = bool(self.agents[0].action[self.agents[0].all_actions_dict_inv['smaller']])
            agent_1_says_smaller = bool(self.agents[1].action[self.agents[1].all_actions_dict_inv['smaller']])
            reward = 0
            if(agent_0_says_larger is first_larger and agent_0_says_smaller is not first_larger):
                if (agent_1_says_larger is not first_larger and agent_1_says_smaller is first_larger):
                    reward = 1
            self.done = True
        # If both agents decided to submit their external representation, submit answer
        if(self.agents[0].is_submitted_ext_repr and self.agents[1].is_submitted_ext_repr):
            self.agents[0].obs = agents[1].submitted_ext_repr
            self.agents[1].obs = agents[0].submitted_ext_repr
            self.time_to_give_an_answer = True

        return next_states_list, reward, self.done

    def select_action(self):
        pass


    def reset(self):
        self.n_agents = len(self.agents)
        self.states = [agent.state for agent in self.agents]
        self.actions = None
        self.time_to_give_an_answer = False
        self.done = False


    def render(self):
        # Concatentate agents images vertically.
        # Each agents image consists of horizontally concatenated images: obs, finger layer, action, external repr.
        img_list = [agent.render() for agent in agents]
        total_img = utils.concat_imgs_v(img_list)
        return total_img
