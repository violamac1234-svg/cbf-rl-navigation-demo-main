"""机械臂多约束 CBF 与二维精确速度投影器。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .kinematics import PlanarArm2D


@dataclass(frozen=True)
class ArmCBFConfig:
    alpha: float = 6.0
    safety_margin: float = 0.025
    q_min: tuple[float, float] = (-2.85, -2.85)
    q_max: tuple[float, float] = (2.85, 2.85)
    velocity_limit: tuple[float, float] = (1.6, 1.6)
    sample_fractions: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass(frozen=True)
class BarrierConstraint:
    """半空间 a @ q_dot >= b；h 是对应安全裕度。"""

    a: np.ndarray
    b: float
    h: float
    label: str


@dataclass(frozen=True)
class FilterResult:
    velocity: np.ndarray
    feasible: bool
    max_violation: float
    active_labels: tuple[str, ...]


def build_barrier_constraints(
    arm: PlanarArm2D,
    q: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    config: ArmCBFConfig,
) -> list[BarrierConstraint]:
    """构造连杆-圆障碍物及关节限位的全部一阶 CBF 约束。"""
    constraints: list[BarrierConstraint] = []
    centers = np.asarray(obstacle_centers, dtype=float).reshape(-1, 2)
    radii = np.asarray(obstacle_radii, dtype=float).reshape(-1)
    if centers.shape[0] != radii.shape[0]:
        raise ValueError("obstacle_centers and obstacle_radii must have equal length")

    for point, jac, point_label in arm.sampled_points(q, config.sample_fractions):
        for obstacle_index, (center, radius) in enumerate(zip(centers, radii)):
            delta = point - center
            distance = float(np.linalg.norm(delta))
            normal = delta / max(distance, 1.0e-9)
            h = distance - (float(radius) + arm.link_radius + config.safety_margin)
            grad_q = normal @ jac
            constraints.append(
                BarrierConstraint(grad_q, -config.alpha * h, h, f"{point_label}/obs{obstacle_index}")
            )

    q = np.asarray(q, dtype=float)
    for joint in range(2):
        unit = np.zeros(2)
        unit[joint] = 1.0
        lower_h = q[joint] - config.q_min[joint]
        upper_h = config.q_max[joint] - q[joint]
        constraints.append(BarrierConstraint(unit.copy(), -config.alpha * lower_h, lower_h, f"q{joint + 1}_min"))
        constraints.append(BarrierConstraint(-unit.copy(), -config.alpha * upper_h, upper_h, f"q{joint + 1}_max"))
    return constraints


def _project_to_halfspaces(u_nom: np.ndarray, a: np.ndarray, b: np.ndarray, tolerance=1.0e-8):
    """解二维 QP：min ||u-u_nom||²/2, s.t. A u >= b。

    二维凸投影的最优点只能是原点、单条边界投影或两条边界交点，
    因而可以枚举求得精确解，不额外依赖 QP 求解库。
    """
    def feasible(candidate):
        return bool(np.all(a @ candidate >= b - tolerance))

    candidates: list[np.ndarray] = []
    if feasible(u_nom):
        candidates.append(u_nom.copy())
    for row, rhs in zip(a, b):
        norm_sq = float(row @ row)
        if norm_sq > 1.0e-12:
            candidate = u_nom + ((rhs - row @ u_nom) / norm_sq) * row
            if feasible(candidate):
                candidates.append(candidate)
    for first, second in combinations(range(len(b)), 2):
        matrix = np.vstack((a[first], a[second]))
        if abs(float(np.linalg.det(matrix))) > 1.0e-10:
            candidate = np.linalg.solve(matrix, np.array([b[first], b[second]]))
            if feasible(candidate):
                candidates.append(candidate)
    if not candidates:
        return np.asarray(u_nom, dtype=float), False
    return min(candidates, key=lambda candidate: float(np.sum((candidate - u_nom) ** 2))), True


def filter_joint_velocity(
    u_nom: np.ndarray,
    constraints: list[BarrierConstraint],
    velocity_limit: tuple[float, float],
) -> FilterResult:
    """把策略关节速度投影到所有 CBF 和速度限幅约束的交集。"""
    limit = np.asarray(velocity_limit, dtype=float)
    rows = [constraint.a for constraint in constraints]
    bounds = [constraint.b for constraint in constraints]
    labels = [constraint.label for constraint in constraints]
    for joint in range(2):
        unit = np.zeros(2)
        unit[joint] = 1.0
        rows.extend((unit.copy(), -unit.copy()))
        bounds.extend((-limit[joint], -limit[joint]))
        labels.extend((f"u{joint + 1}_min", f"u{joint + 1}_max"))
    a = np.asarray(rows, dtype=float)
    b = np.asarray(bounds, dtype=float)
    velocity, feasible = _project_to_halfspaces(np.asarray(u_nom, dtype=float), a, b)
    violations = b - a @ velocity
    active = tuple(label for label, residual in zip(labels, a @ velocity - b) if abs(float(residual)) < 1.0e-5)
    return FilterResult(velocity, feasible, float(max(0.0, np.max(violations))), active)
