"""二维二连杆机械臂运动学（NumPy，无仿真器依赖）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanarArm2D:
    """平面二连杆；状态为关节角 q，控制量为关节速度 q_dot。"""

    link_lengths: tuple[float, float] = (1.0, 0.8)
    link_radius: float = 0.055

    def joint_positions(self, q: np.ndarray) -> np.ndarray:
        """返回 base、elbow、end-effector 三点，形状为 (3, 2)。"""
        q1, q2 = np.asarray(q, dtype=float)
        l1, l2 = self.link_lengths
        elbow = l1 * np.array([np.cos(q1), np.sin(q1)])
        tip = elbow + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
        return np.vstack((np.zeros(2), elbow, tip))

    def point_and_jacobian(self, q: np.ndarray, link: int, fraction: float) -> tuple[np.ndarray, np.ndarray]:
        """返回某连杆上一点的位置及对 q 的 2x2 雅可比。

        ``link`` 为 0/1，``fraction`` 从近端 0 到远端 1。
        """
        if link not in (0, 1) or not 0.0 <= fraction <= 1.0:
            raise ValueError("link must be 0 or 1 and fraction must be in [0, 1]")
        q1, q2 = np.asarray(q, dtype=float)
        l1, l2 = self.link_lengths
        perp1 = np.array([-np.sin(q1), np.cos(q1)])
        if link == 0:
            point = fraction * l1 * np.array([np.cos(q1), np.sin(q1)])
            jac = np.column_stack((fraction * l1 * perp1, np.zeros(2)))
            return point, jac

        angle12 = q1 + q2
        perp12 = np.array([-np.sin(angle12), np.cos(angle12)])
        elbow = l1 * np.array([np.cos(q1), np.sin(q1)])
        point = elbow + fraction * l2 * np.array([np.cos(angle12), np.sin(angle12)])
        jac = np.column_stack((l1 * perp1 + fraction * l2 * perp12, fraction * l2 * perp12))
        return point, jac

    def end_effector_jacobian(self, q: np.ndarray) -> np.ndarray:
        """末端执行器雅可比。"""
        return self.point_and_jacobian(q, link=1, fraction=1.0)[1]

    def sampled_points(self, q: np.ndarray, fractions=(0.2, 0.4, 0.6, 0.8, 1.0)):
        """依次产生两根连杆的碰撞采样点、雅可比和标签。"""
        for link in (0, 1):
            for fraction in fractions:
                point, jac = self.point_and_jacobian(q, link, float(fraction))
                yield point, jac, f"link{link + 1}@{fraction:.1f}"
