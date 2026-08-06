import unittest

import torch

from experiment_configs import TABLE_III_CONFIGS, TRAINING_METHODS
from nav_env.unified_navigation_env import UnifiedNavigationEnv


class ExperimentConfigTests(unittest.TestCase):
    def test_table_iii_has_all_twelve_configurations(self):
        self.assertEqual(len(TABLE_III_CONFIGS), 12)
        self.assertEqual(set(TRAINING_METHODS), {"nominal", "reward_only", "filter_only", "dual"})


class CbfTests(unittest.TestCase):
    """验证 CBF 梯度方向、闭式过滤条件和复现指标接口。"""
    def setUp(self):
        self.env = UnifiedNavigationEnv(
            num_envs=1,
            num_obstacles=1,
            device="cpu",
            noise_level=0.0,
            max_episode_steps=20,
        )
        self.env._obstacle_positions[:] = torch.tensor([[[5.0, 5.0]]])
        self.env._obstacle_radii[:] = 0.5

    def tearDown(self):
        self.env.close()

    def test_obstacle_gradient_points_away_from_obstacle(self):
        position = torch.tensor([[6.0, 5.0]])
        gradient = self.env.gradient_h_function(position, self.env._obstacle_positions)
        torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0]]))

    def test_filter_satisfies_active_cbf_constraint(self):
        # 机器人位于障碍物右侧，向左动作会逼近障碍物，过滤后必须满足 psi >= 0。
        position = torch.tensor([[6.0, 5.0]])
        unsafe_velocity = torch.tensor([[-1.0, 0.0]])
        filtered, _ = self.env.filter_velocity(
            position, unsafe_velocity, self.env._obstacle_positions, self.env._obstacle_radii
        )
        h = self.env.h_function(position, self.env._obstacle_positions, self.env._obstacle_radii)
        grad = self.env.gradient_h_function(position, self.env._obstacle_positions)
        post_condition = torch.sum(grad * filtered, dim=1) + self.env.cbf_alpha * h
        self.assertGreaterEqual(post_condition.item(), -1e-6)

    def test_step_reports_reproduction_metrics(self):
        _, _, _, extras = self.env.step(torch.zeros((1, 2)))
        for key in (
            "cbf_violated",
            "filter_activated",
            "action_correction_norm",
            "min_safety_margin",
        ):
            self.assertIn(key, extras["log"])

    def test_observation_dimension_matches_closest_obstacle_design(self):
        obs, _ = self.env.get_observations()
        self.assertEqual(obs.shape, (1, 9))


if __name__ == "__main__":
    unittest.main()
