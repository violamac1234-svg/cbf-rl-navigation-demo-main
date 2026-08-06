import unittest

import torch

from isaaclab_planar_avoidance.cbf_core import cbf_reward, circular_barrier, closed_form_filter


class TestPlanarCBF(unittest.TestCase):
    def test_gradient_points_away_from_obstacle(self):
        h, grad = circular_barrier(
            torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 0.0]]), 0.2, 0.3
        )
        torch.testing.assert_close(h, torch.tensor([0.5]))
        torch.testing.assert_close(grad, torch.tensor([[1.0, 0.0]]))

    def test_filter_leaves_safe_velocity_unchanged(self):
        velocity = torch.tensor([[0.2, 0.0]])
        safe, active, psi = closed_form_filter(
            velocity, torch.tensor([0.5]), torch.tensor([[1.0, 0.0]]), 5.0
        )
        torch.testing.assert_close(safe, velocity)
        self.assertFalse(active.item())
        self.assertGreater(psi.item(), 0.0)

    def test_filter_projects_unsafe_velocity_to_boundary(self):
        velocity = torch.tensor([[-1.0, 0.2]])
        h = torch.tensor([0.1])
        grad = torch.tensor([[1.0, 0.0]])
        safe, active, _ = closed_form_filter(velocity, h, grad, alpha=5.0)
        self.assertTrue(active.item())
        condition = (grad * safe).sum(dim=-1) + 5.0 * h
        torch.testing.assert_close(condition, torch.zeros_like(condition), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(safe, torch.tensor([[-0.5, 0.2]]))

    def test_closest_obstacle_is_selected_per_environment(self):
        robot = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
        obstacles = torch.tensor([
            [[2.0, 0.0], [0.0, 1.0]],
            [[-0.8, 0.0], [0.0, 3.0]],
        ])
        _, grad = circular_barrier(robot, obstacles, 0.2, 0.2)
        torch.testing.assert_close(grad, torch.tensor([[0.0, -1.0], [1.0, 0.0]]))

    def test_cbf_reward_is_zero_without_intervention(self):
        velocity = torch.tensor([[0.2, 0.0]])
        reward = cbf_reward(velocity, velocity, torch.tensor([1.0]))
        torch.testing.assert_close(reward, torch.zeros_like(reward))


if __name__ == "__main__":
    unittest.main()
