import itertools
import unittest

import numpy as np

from src.SingleAgentEnv import SingleAgentEnv
from src.Reward import Reward


env_params = {
                'reward': Reward(**{'bad_label_punishment': False,
                                    'curiosity': False,
                                    'time_penalty': False}),
                'max_CL_objects': 3,
                'CL_phases': 1,
                'max_episode_objects': 2,
                'obs_dim': 14,
                'n_actions': 1,
                'max_episode_length': 1,
                'n_episodes_per_phase': 1,
                'max_object_size': 2,
            }


class TestSingleAgentEnv(unittest.TestCase):
    """
    This class contains unit tests for the Single Agent Environment.
    """
    def test_detect_adjacent_objects(self):
        """
        This method tests that the control for objects adjacency works.
        """
        # generate a random-sized scene
        scene_size = np.random.randint(4, 10)
        print(f'{scene_size=}')

        # generate a random-sized object
        object_size = np.random.randint(2, scene_size - 1)
        print(f"{object_size=}")

        # generate a random position for the object
        upper_left_point = (np.random.randint(0, scene_size + 1 - object_size),
                            np.random.randint(0, scene_size + 1 - object_size))
        print(f"{upper_left_point=}")

        # generate object coordinates
        object_coordinates = set(itertools.product(range(upper_left_point[0], upper_left_point[0] + object_size),
                                                   range(upper_left_point[1], upper_left_point[1] + object_size)))

        # generate a (1,1)-shaped new object
        # that is adjacent to the first one
        valid_point = False
        new_point = (0, 0)
        print(f"{object_coordinates=}")

        while not valid_point:
            new_point = (np.random.randint(0, scene_size),
                         np.random.randint(0, scene_size))

            print(f"{new_point=}")

            if new_point not in object_coordinates:
                (x, y) = new_point
                if (
                        ((x + 1, y) in object_coordinates) or
                        ((x, y + 1) in object_coordinates) or
                        ((x - 1, y) in object_coordinates) or
                        ((x, y - 1) in object_coordinates)
                ):
                    valid_point = True

        check_result = SingleAgentEnv._check_squares_intersection_adjacency({new_point}, set(object_coordinates))

        self.assertEqual(True, check_result)

    def test_new_object_is_squared(self):
        env = SingleAgentEnv(**env_params)

        random_size = np.random.randint(env.obs_dim) + 1
        print(f"{random_size=}")

        coordinates = env._generate_square(size=random_size,
                                           picture_objects_coordinates=set())
        print(f"{coordinates=}")

        self.assertEqual(random_size**2, len(coordinates))

if __name__ == '__main__':
    unittest.main()
