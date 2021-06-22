"""
The purpose of this class is to test the Reward class, and in
particular its core get_reward method.
"""
import random
import unittest

from src.Reward import Reward
from src.MockEnvironment import MockEnvironment


class TestReward(unittest.TestCase):

    def test_right_label_reward(self):
        mock_env = MockEnvironment()

        reward_object = Reward(
            bad_label_punishment=False,
            curiosity=False,
            time_penalty=0,
        )

        reward_value, __ = reward_object.get_reward(
            env=mock_env,
            action=len(mock_env.actions) + mock_env.obs_label - 1,
            visit_history={},
        )

        self.assertEqual(reward_value, 1)

    def test_wrong_label_reward(self):
        mock_env = MockEnvironment()

        reward_object = Reward(
            bad_label_punishment=False,
            curiosity=False,
            time_penalty=0,
        )

        reward_value, __ = reward_object.get_reward(
            env=mock_env,
            action=len(mock_env.actions) + mock_env.obs_label,
            visit_history={},
        )

        self.assertEqual(reward_value, 0)

    def test_right_label_wrong_time(self):
        mock_env = MockEnvironment()
        mock_env.first_label_output = False

        reward_object = Reward(
            bad_label_punishment=False,
            curiosity=False,
            time_penalty=0,
        )

        reward_value, __ = reward_object.get_reward(
            env=mock_env,
            action=len(mock_env.actions) + mock_env.obs_label,
            visit_history={},
        )

        self.assertEqual(reward_value, 0)


if __name__ == '__main__':
    unittest.main()
