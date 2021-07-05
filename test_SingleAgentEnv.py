import itertools
import random
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
    @staticmethod
    def _generate_random_scene_setting(scene_size: int = None,
                                       object_size: int = None) -> tuple:
        """
        This method is used in the tests for the object generation
        process. It creates the parameters describing a random scene:
        the random size of a scene, the random size of an object
        in the scene, the random position of the object.
        If scene_size is passed, the method can also be used to
        generate new object parameters within the same scene.

        Args:
            scene_size: size of an existing scene (optional).
            object_size: size of the object generated in the scene (optional).

        Returns:
            The values of the scene size, object size, and object
            position.
        """
        # generate a random-sized scene
        if scene_size is None:
            scene_size = np.random.randint(4, 10)
        print(f'{scene_size=}')

        # generate a random-sized object
        if object_size is None:
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

    def test_detect_overlapping_objects(self):
        """
        This method tests that overlapping objects detection works.
        """
        scene_size, object_size, upper_left_point = self._generate_random_scene_setting()

        # generate object coordinates
        object_coordinates = self._generate_object_coordinates(upper_left_point, object_size)

        valid_object = False
        new_object_coordinates = {}
        while not valid_object:
            # generate a random-sized object
            # that is overlapping with the first one
            __, object_size, upper_left_point = self._generate_random_scene_setting(scene_size)

            new_object_coordinates = self._generate_object_coordinates(upper_left_point, object_size)

            if len(new_object_coordinates & object_coordinates) > 0:
                valid_object = True

        check_result = SingleAgentEnv._check_squares_intersection_adjacency(new_object_coordinates, object_coordinates)

        self.assertEqual(True, check_result)

    def test_new_object_is_squared(self):
        """This method tests "heuristically" that a new object is
        squared."""
        env = SingleAgentEnv(**env_params)

        random_size = np.random.randint(env.obs_dim) + 1
        print(f"{random_size=}")

        coordinates = env._generate_square(size=random_size,
                                           picture_objects_coordinates=set())
        print(f"{coordinates=}")

        self.assertEqual(random_size**2, len(coordinates))

    def test_detect_impossible_overlapping_fit(self):
        """This method tests that an impossible scene setting due to
        a new object being necessarily overlapping with existing
        objects is successfully spotted and thus avoided.
        """
        env = SingleAgentEnv(**env_params)

        scene_size, object_size, upper_left_point = self._generate_random_scene_setting()
        env.obs_dim = scene_size  # the tested method depends on this parameter

        # generate object coordinates
        object_coordinates = self._generate_object_coordinates(upper_left_point, object_size)

        # generate coordinates for a new object
        # that for sure overlaps the existing one
        new_object_size = scene_size - object_size + 1
        
        check_result = env._check_square_can_fit(new_object_size, object_coordinates)
        
        self.assertEqual(False, check_result)

    def test_detect_impossible_adjacent_fit(self):
        """This method tests that an impossible scene setting due to
        a new object being necessarily adjacent to existing ones
        is successfully spotted and thus avoided.
        """
        env = SingleAgentEnv(**env_params)

        # generate a scene with a square in the middle
        scene_size = np.random.randint(3, 10)
        env.obs_dim = scene_size  # the tested method depends on this attribute
        object_diagonal_position = np.random.randint(1, scene_size // 2)  # border excluded
        upper_left_point = (object_diagonal_position, object_diagonal_position)

        object_size = 1
        __, y = upper_left_point
        while y < scene_size - (upper_left_point[0] + 1):
            object_size += 1
            y += 1

        # generate object coordinates
        object_coordinates = self._generate_object_coordinates(upper_left_point, object_size)
        SingleAgentEnv._plot_scene(scene_size=scene_size, coordinates=object_coordinates)

        # the new object will be sized as the
        # space between old object and border
        new_object_size = upper_left_point[0]
        print(new_object_size)

        # fill corners (they are the only spots
        # where the new object can be placed legally)
        # we start from the upper left corner of the scene
        # and we fill every corner; the displacement is equal
        # to the size of the main object + the size of the border
        object_coordinates |= self._generate_object_coordinates((0, 0), new_object_size)
        object_coordinates |= self._generate_object_coordinates((object_size + upper_left_point[0], 0), new_object_size)
        object_coordinates |= self._generate_object_coordinates((0, object_size + upper_left_point[0]), new_object_size)
        object_coordinates |= self._generate_object_coordinates((object_size + upper_left_point[0],
                                                                 object_size + upper_left_point[0]), new_object_size)
        SingleAgentEnv._plot_scene(scene_size=scene_size, coordinates=object_coordinates)

        check_result = env._check_square_can_fit(new_object_size, object_coordinates)

        self.assertEqual(False, check_result)


if __name__ == '__main__':
    unittest.main()
