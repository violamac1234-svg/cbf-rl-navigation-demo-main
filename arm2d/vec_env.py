"""可在 GPU 上并行训练的二维二连杆关节速度环境。"""

from __future__ import annotations

from itertools import combinations

import torch
from rsl_rl.env.vec_env import VecEnv

from experiment_configs import MethodConfig


class Arm2DVecEnv(VecEnv):
    """低阶机械臂环境：策略输出关节速度，状态由显式欧拉法更新。

    障碍物由圆表示；两根连杆各取 5 个点建立 CBF。安全过滤器在二维动作空间
    枚举半空间边界投影和两两交点，从而批量求解精确的欧氏投影 QP。
    """

    def __init__(
        self,
        num_envs: int,
        method: MethodConfig,
        device: str = "cuda",
        seed: int = 42,
        max_episode_steps: int = 320,
        dt: float = 0.025,
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.num_actions = 2
        self.device = torch.device(device)
        self.max_episode_length = int(max_episode_steps)
        self.dt = float(dt)
        self.cfg = {"method": method}
        self.use_filter = method.use_training_filter
        self.use_cbf_reward = method.use_cbf_reward

        self.link_lengths = torch.tensor([1.0, 0.8], device=self.device)
        self.link_radius = 0.055
        self.safety_margin = 0.025
        self.alpha = 6.0
        self.q_min = torch.tensor([-2.85, -2.85], device=self.device)
        self.q_max = torch.tensor([2.85, 2.85], device=self.device)
        self.velocity_limit = torch.tensor([1.6, 1.6], device=self.device)
        self.goal_radius = 0.075
        self.reward_cbf_weight = 5.0
        self.reward_cbf_sigma = 0.5
        self.fractions = torch.linspace(0.2, 1.0, 5, device=self.device)

        # QP 中固定有 10 条障碍物约束、4 条关节约束和 4 条速度约束。
        self._pair_indices = torch.tensor(
            list(combinations(range(18), 2)), dtype=torch.long, device=self.device
        )
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(seed)

        self.q = torch.zeros((self.num_envs, 2), device=self.device)
        self.last_velocity = torch.zeros_like(self.q)
        self.goal = torch.zeros_like(self.q)
        self.obstacle_center = torch.zeros_like(self.q)
        self.obstacle_radius = torch.zeros(self.num_envs, device=self.device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset()

    def _rand(self, shape):
        return torch.rand(shape, device=self.device, generator=self._generator)

    def _kinematics(self, q: torch.Tensor):
        q1, q12 = q[:, 0], q.sum(dim=1)
        l1, l2 = self.link_lengths
        elbow = l1 * torch.stack((torch.cos(q1), torch.sin(q1)), dim=1)
        tip = elbow + l2 * torch.stack((torch.cos(q12), torch.sin(q12)), dim=1)
        return elbow, tip

    def _points_and_jacobians(self, q: torch.Tensor):
        """返回 10 个碰撞点及其雅可比，形状为 N×10×2 和 N×10×2×2。"""
        count = q.shape[0]
        q1, q12 = q[:, 0], q.sum(dim=1)
        l1, l2 = self.link_lengths
        direction1 = torch.stack((torch.cos(q1), torch.sin(q1)), dim=1)
        perp1 = torch.stack((-torch.sin(q1), torch.cos(q1)), dim=1)
        direction12 = torch.stack((torch.cos(q12), torch.sin(q12)), dim=1)
        perp12 = torch.stack((-torch.sin(q12), torch.cos(q12)), dim=1)
        fractions = self.fractions.view(1, -1, 1)

        points1 = fractions * l1 * direction1[:, None, :]
        jac1 = torch.zeros((count, 5, 2, 2), device=self.device)
        jac1[:, :, :, 0] = fractions * l1 * perp1[:, None, :]

        elbow = l1 * direction1
        points2 = elbow[:, None, :] + fractions * l2 * direction12[:, None, :]
        jac2 = torch.zeros_like(jac1)
        jac2[:, :, :, 0] = l1 * perp1[:, None, :] + fractions * l2 * perp12[:, None, :]
        jac2[:, :, :, 1] = fractions * l2 * perp12[:, None, :]
        return torch.cat((points1, points2), dim=1), torch.cat((jac1, jac2), dim=1)

    def _obstacle_barriers(self, q: torch.Tensor):
        points, jacobians = self._points_and_jacobians(q)
        delta = points - self.obstacle_center[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(1.0e-8)
        h = distance - (self.obstacle_radius[:, None] + self.link_radius + self.safety_margin)
        normal = delta / distance[:, :, None]
        grad = torch.einsum("nki,nkij->nkj", normal, jacobians)
        return h, grad

    def _all_constraints(self):
        h_obs, grad_obs = self._obstacle_barriers(self.q)
        eye = torch.eye(2, device=self.device).expand(self.num_envs, -1, -1)
        h_lower = self.q - self.q_min
        h_upper = self.q_max - self.q
        a_cbf = torch.cat((grad_obs, eye, -eye), dim=1)
        h_cbf = torch.cat((h_obs, h_lower, h_upper), dim=1)
        b_cbf = -self.alpha * h_cbf

        # |q_dot| <= velocity_limit 也写成 A u >= b。
        a_velocity = torch.cat((eye, -eye), dim=1)
        b_velocity = -self.velocity_limit.repeat(2).expand(self.num_envs, -1)
        return torch.cat((a_cbf, a_velocity), dim=1), torch.cat((b_cbf, b_velocity), dim=1), h_obs

    def _filter(self, nominal: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
        """批量二维 QP；返回投影速度、是否找到可行点及过滤前最小 psi。"""
        n, m, _ = a.shape
        residual_nominal = torch.einsum("nmd,nd->nm", a, nominal) - b
        psi_min = residual_nominal[:, :14].min(dim=1).values

        norm_sq = (a * a).sum(dim=2).clamp_min(1.0e-12)
        correction = (-residual_nominal / norm_sq)[:, :, None] * a
        single = nominal[:, None, :] + correction

        first, second = self._pair_indices[:, 0], self._pair_indices[:, 1]
        a1, a2 = a[:, first], a[:, second]
        b1, b2 = b[:, first], b[:, second]
        determinant = a1[:, :, 0] * a2[:, :, 1] - a1[:, :, 1] * a2[:, :, 0]
        valid_det = determinant.abs() > 1.0e-9
        safe_det = torch.where(valid_det, determinant, torch.ones_like(determinant))
        x = (b1 * a2[:, :, 1] - a1[:, :, 1] * b2) / safe_det
        y = (a1[:, :, 0] * b2 - b1 * a2[:, :, 0]) / safe_det
        pair = torch.stack((x, y), dim=2)

        candidates = torch.cat((nominal[:, None, :], single, pair), dim=1)
        feasible = torch.einsum("nmd,ncd->ncm", a, candidates) >= b[:, None, :] - 1.0e-6
        feasible = feasible.all(dim=2)
        feasible[:, 1 + m:] &= valid_det
        costs = ((candidates - nominal[:, None, :]) ** 2).sum(dim=2)
        costs = costs.masked_fill(~feasible, torch.inf)
        best_cost, best_index = costs.min(dim=1)
        batch = torch.arange(n, device=self.device)
        filtered = candidates[batch, best_index]
        found = torch.isfinite(best_cost)
        filtered = torch.where(found[:, None], filtered, torch.zeros_like(filtered))
        return filtered, found, psi_min

    def _observations(self):
        _, tip = self._kinematics(self.q)
        return torch.cat(
            (
                torch.sin(self.q),
                torch.cos(self.q),
                self.last_velocity / self.velocity_limit,
                (self.goal - tip) / 1.8,
                (self.obstacle_center - tip) / 1.8,
                self.obstacle_radius[:, None] / 0.25,
            ),
            dim=1,
        )

    def get_observations(self):
        return self._observations(), {"observations": {}}

    def _reset_indices(self, indices: torch.Tensor):
        """生成起点、可达目标和位于末端直线路径附近的单个障碍物。"""
        if indices.numel() == 0:
            return
        count = indices.numel()
        q0 = torch.empty((count, 2), device=self.device)
        q_goal = torch.empty_like(q0)
        centers = torch.empty_like(q0)
        radii = torch.empty(count, device=self.device)
        accepted = torch.zeros(count, dtype=torch.bool, device=self.device)
        for _ in range(80):
            pending = ~accepted
            if not pending.any():
                break
            k = int(pending.sum())
            candidate_q0 = torch.stack(
                (-1.15 + 0.7 * self._rand(k), 0.45 + 1.0 * self._rand(k)), dim=1
            )
            candidate_goal_q = torch.stack(
                (0.25 + 0.95 * self._rand(k), -1.35 + 0.9 * self._rand(k)), dim=1
            )
            _, start_tip = self._kinematics(candidate_q0)
            _, goal_tip = self._kinematics(candidate_goal_q)
            path = goal_tip - start_tip
            path_length = torch.linalg.vector_norm(path, dim=1)
            perpendicular = torch.stack((-path[:, 1], path[:, 0]), dim=1) / path_length.clamp_min(1e-6)[:, None]
            offset = (self._rand(k) - 0.5) * 0.22
            candidate_center = start_tip + (0.40 + 0.20 * self._rand(k))[:, None] * path + offset[:, None] * perpendicular
            candidate_radius = 0.13 + 0.05 * self._rand(k)

            # 临时计算候选起点/终点的点到障碍物裕度。
            saved_center, saved_radius = self.obstacle_center[indices[pending]].clone(), self.obstacle_radius[indices[pending]].clone()
            self.obstacle_center[indices[pending]] = candidate_center
            self.obstacle_radius[indices[pending]] = candidate_radius
            old_q = self.q[indices[pending]].clone()
            self.q[indices[pending]] = candidate_q0
            start_h, _ = self._obstacle_barriers(self.q[indices[pending]]) if k == self.num_envs else self._barriers_for(candidate_q0, candidate_center, candidate_radius)
            goal_h, _ = self._barriers_for(candidate_goal_q, candidate_center, candidate_radius)
            self.q[indices[pending]] = old_q
            self.obstacle_center[indices[pending]] = saved_center
            self.obstacle_radius[indices[pending]] = saved_radius
            valid = (path_length > 0.65) & (start_h.min(dim=1).values > 0.10) & (goal_h.min(dim=1).values > 0.08)
            pending_positions = torch.where(pending)[0]
            chosen = pending_positions[valid]
            q0[chosen] = candidate_q0[valid]
            q_goal[chosen] = candidate_goal_q[valid]
            centers[chosen] = candidate_center[valid]
            radii[chosen] = candidate_radius[valid]
            accepted[chosen] = True
        if not accepted.all():
            raise RuntimeError("Failed to sample collision-free arm tasks")
        _, goals = self._kinematics(q_goal)
        self.q[indices] = q0
        self.goal[indices] = goals
        self.obstacle_center[indices] = centers
        self.obstacle_radius[indices] = radii
        self.last_velocity[indices] = 0.0
        self.episode_length_buf[indices] = 0

    def _barriers_for(self, q, center, radius):
        points, jacobians = self._points_and_jacobians(q)
        delta = points - center[:, None, :]
        distance = torch.linalg.vector_norm(delta, dim=2).clamp_min(1.0e-8)
        h = distance - (radius[:, None] + self.link_radius + self.safety_margin)
        normal = delta / distance[:, :, None]
        return h, torch.einsum("nki,nkij->nkj", normal, jacobians)

    def reset(self):
        self._reset_indices(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()

    def step(self, actions: torch.Tensor):
        actions = torch.clamp(actions.to(self.device), -self.velocity_limit, self.velocity_limit)
        _, old_tip = self._kinematics(self.q)
        old_distance = torch.linalg.vector_norm(self.goal - old_tip, dim=1)
        a, b, _ = self._all_constraints()
        safe_actions, qp_feasible, psi_min = self._filter(actions, a, b)
        executed = safe_actions if self.use_filter else actions
        self.last_velocity = executed
        self.q = torch.clamp(self.q + self.dt * executed, self.q_min, self.q_max)
        self.episode_length_buf += 1

        _, tip = self._kinematics(self.q)
        distance = torch.linalg.vector_norm(self.goal - tip, dim=1)
        h_obs, _ = self._obstacle_barriers(self.q)
        minimum_margin = h_obs.min(dim=1).values
        physical_margin = minimum_margin + self.safety_margin
        success = distance < self.goal_radius
        collision = physical_margin < 0.0
        timeout = self.episode_length_buf >= self.max_episode_length
        done = success | collision | timeout

        progress = (old_distance - distance) / (self.dt * 1.8 * self.velocity_limit.max())
        correction_sq = ((actions - safe_actions) ** 2).sum(dim=1)
        if self.use_cbf_reward:
            cbf_reward = self.reward_cbf_weight * (
                torch.minimum(psi_min, torch.zeros_like(psi_min))
                + torch.exp(-correction_sq / self.reward_cbf_sigma**2)
                - 1.0
            )
        else:
            cbf_reward = torch.zeros_like(distance)
        reward = 4.0 * progress - 0.005 + 12.0 * success.float() - 12.0 * collision.float() + cbf_reward

        extras = {
            "observations": {},
            "time_outs": timeout,
            "log": {
                "arm/success": success.float(),
                "arm/collision": collision.float(),
                "arm/distance_to_goal": distance,
                "arm/min_safety_margin": minimum_margin,
                "arm/filter_activated": (correction_sq > 1.0e-10).float(),
                "arm/action_correction": torch.sqrt(correction_sq),
                "arm/cbf_reward": cbf_reward,
                "arm/qp_feasible": qp_feasible.float(),
            },
        }
        reset_ids = torch.where(done)[0]
        self._reset_indices(reset_ids)
        obs = self._observations()
        return obs, reward, done, extras
