import numpy as np


class Reward:
    def __init__(self, reward_parameters: dict):
        self.parameters = reward_parameters

    def get_reward(self, env: src.SingleAgentEnv, action: int, visit_history: dict) -> float:
        """Reward function. Currently implementing only label-based reward
        logic, giving positive reward when the action is a label action
        which corresponds to the correct label. We can also give a negative
        reward to the agent when it outputs the wrong label.
        We can implement a simple curiosity mechanism, saving
        the number of times the agent has been in each different state.
        """
        n_actions = len(env.actions_dict)
        if env.CL:
            start_labels_actions = n_actions - env.max_CL_objects
        else:
            start_labels_actions = n_actions - env.max_objects

        reward = 0

        if action in range(start_labels_actions, n_actions):
            chosen_label = int(env.actions_dict[action])
            true_label = np.argmax(env.obs_label) + 1

            if chosen_label == true_label:
                reward = 1
            elif self.parameters['bad_label_punishment']:
                reward = -.5

        # implementing a simple curiosity mechanism
        if self.parameters['curiosity']:
            current_state_hash = env.get_state_hash()
            n_visits = visit_history.get(current_state_hash, 0)
            if n_visits == 0:
                reward += .1

            visit_history.setdefault(current_state_hash, 0)
            visit_history[current_state_hash] += 1

        return reward

        # in case we want to keep label distance-based rewards...
        # label_dist = self.compare_labels(label_slice, self.obs_label)

        # reward based on scene finger position
        # finger_index = self.fingerlayer_scene.fingerlayer.argmax()
        # finger_position = np.unravel_index(finger_index, self.fingerlayer_scene.fingerlayer.shape)

        # if self.obs[finger_position] == 1:
        # reward += 0.1 # TODO: diminishing reward?

        # TODO: reward diminishing with time

        # TODO: reward showing how to create repr. for small quantities