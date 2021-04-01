import utils

class MultiAgentEnvironment():
    '''
    Implements Multiagent environment that looks like Single-Agent environment from the outside:
    usual attributes of RL environments are lists,
    e.g. env.action=[agent_1_action, agent_2_action],
         env.state=[agent_1_state, agent_2_state],
    '''
    def __init__(self, agent_list):
        self.agents = agent_list
        self.n_agents = len(agent_list)
        self.states = [agent.state for agent in agents]
        self.actions = None

    def step(self, actions):
        reward = 0
        next_states_list = []
        for i in range(self.n_agents):
            next_state = self.agents[i].step(actions[i])
            next_states_list.append(next_state)

        # If both agents decided to submit their external representation, submit answer
        if(agents[0].submit_external_repr and agents[1].submit_external_repr):


        return reward, next_states_list

    def render(self):
        # Concatentate agents images vertically.
        # Each agents image consists of horizontally concatenated images: obs, finger layer, action, external repr.
        img_list = [agent.render() for agent in agents]
        total_img = utils.concat_imgs_v(img_list)
        return total_img
