import random

class MockEnvironment:

    def __init__(self):
        self.CL = True
        self.obs_label = random.randint(1, 10)
        self.max_CL_objects = self.obs_label
        self.first_label_output = True
        self.actions = ['mod_point', 'left', 'right', 'up', 'down', 'left', 'right', 'up', 'down']

        self.actions_dict = dict(zip(
            range(len(self.actions)),
            self.actions
        ))

        for n in range(self.max_CL_objects):
            n_keys = len(self.actions_dict)
            self.actions_dict.setdefault(n_keys, n+1)

    @staticmethod
    def get_state_hash():
        return random.randint()
