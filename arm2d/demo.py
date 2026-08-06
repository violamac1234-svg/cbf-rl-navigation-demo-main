"""先于强化学习验证二维二连杆 CBF 过滤器，并生成对比图。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np

from .cbf import ArmCBFConfig, build_barrier_constraints, filter_joint_velocity
from .kinematics import PlanarArm2D


# Windows 默认 Matplotlib 字体不含中文；显式注册黑体，保证输出图片标题可读。
_CHINESE_FONT = Path("C:/Windows/Fonts/simhei.ttf")
if _CHINESE_FONT.exists():
    font_manager.fontManager.addfont(_CHINESE_FONT)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=_CHINESE_FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Rollout:
    q: np.ndarray
    tip: np.ndarray
    minimum_margin: np.ndarray
    filter_active: np.ndarray
    reached_goal: bool


def nominal_controller(arm: PlanarArm2D, q: np.ndarray, goal: np.ndarray, velocity_limit: np.ndarray) -> np.ndarray:
    """末端笛卡尔误差经阻尼伪逆映射为关节速度。"""
    error = goal - arm.joint_positions(q)[-1]
    desired_tip_velocity = 2.2 * error
    speed = np.linalg.norm(desired_tip_velocity)
    if speed > 1.0:
        desired_tip_velocity /= speed
    jac = arm.end_effector_jacobian(q)
    damping = 0.08
    u = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(2), desired_tip_velocity)
    return np.clip(u, -velocity_limit, velocity_limit)


def rollout(filtered: bool, q0: np.ndarray, goal: np.ndarray, centers: np.ndarray, radii: np.ndarray) -> Rollout:
    arm = PlanarArm2D()
    config = ArmCBFConfig()
    dt, steps = 0.025, 520
    q = q0.copy()
    q_history, tip_history, margins, active_history = [], [], [], []
    reached = False
    for _ in range(steps):
        constraints = build_barrier_constraints(arm, q, centers, radii, config)
        u_nom = nominal_controller(arm, q, goal, np.asarray(config.velocity_limit))
        result = filter_joint_velocity(u_nom, constraints, config.velocity_limit)
        u = result.velocity if filtered else u_nom
        q_history.append(q.copy())
        tip_history.append(arm.joint_positions(q)[-1])
        margins.append(min(c.h for c in constraints if "/obs" in c.label))
        active_history.append(filtered and np.linalg.norm(result.velocity - u_nom) > 1.0e-6)
        q = np.clip(q + dt * u, config.q_min, config.q_max)
        if np.linalg.norm(arm.joint_positions(q)[-1] - goal) < 0.035:
            reached = True
            break
    return Rollout(np.asarray(q_history), np.asarray(tip_history), np.asarray(margins), np.asarray(active_history), reached)


def _draw_rollout(axis, rollout_data: Rollout, title: str, arm: PlanarArm2D, goal, centers, radii):
    for center, radius in zip(centers, radii):
        axis.add_patch(Circle(center, radius, color="#d95f5f", alpha=0.82))
        axis.add_patch(Circle(center, radius + arm.link_radius + 0.025, fill=False, color="#8f2525", ls="--", lw=1))
    indices = np.unique(np.linspace(0, len(rollout_data.q) - 1, 13).astype(int))
    for index in indices:
        points = arm.joint_positions(rollout_data.q[index])
        axis.plot(points[:, 0], points[:, 1], color="#5b7691", alpha=0.18, lw=3)
    final_points = arm.joint_positions(rollout_data.q[-1])
    axis.plot(final_points[:, 0], final_points[:, 1], "o-", color="#244a69", lw=4, ms=6)
    axis.plot(rollout_data.tip[:, 0], rollout_data.tip[:, 1], color="#e28b27", lw=2.2, label="末端轨迹")
    axis.scatter(*rollout_data.tip[0], s=55, color="#202020", zorder=5, label="起点")
    axis.scatter(*goal, s=130, marker="*", color="#2d9b58", edgecolor="white", zorder=6, label="目标")
    axis.set(title=title, xlim=(-0.25, 1.95), ylim=(-1.25, 1.65), xlabel="x (m)", ylabel="y (m)")
    axis.set_aspect("equal")
    axis.grid(alpha=0.2)


def make_figure(output: Path) -> dict[str, float | bool]:
    arm = PlanarArm2D()
    q0 = np.array([-0.78, 1.15])
    goal = np.array([0.35, 1.35])
    centers = np.array([[0.88, 0.28]])
    radii = np.array([0.24])
    nominal = rollout(False, q0, goal, centers, radii)
    safe = rollout(True, q0, goal, centers, radii)

    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    _draw_rollout(axes[0], nominal, "Nominal（不过滤）", arm, goal, centers, radii)
    _draw_rollout(axes[1], safe, "CBF-filter", arm, goal, centers, radii)
    axes[1].legend(loc="lower left", fontsize=8)
    axes[2].plot(np.arange(len(nominal.minimum_margin)) * 0.025, nominal.minimum_margin, label="Nominal", lw=2)
    axes[2].plot(np.arange(len(safe.minimum_margin)) * 0.025, safe.minimum_margin, label="CBF-filter", lw=2)
    axes[2].axhline(0, color="#b53232", ls="--", lw=1.2, label="碰撞边界")
    axes[2].fill_between(np.arange(len(safe.filter_active)) * 0.025, -0.15, 0.5, where=safe.filter_active,
                         color="#4c9f70", alpha=0.09, label="过滤器介入")
    axes[2].set(title="最小安全裕度", xlabel="时间 (s)", ylabel="h (m)", ylim=(-0.15, 0.5))
    axes[2].grid(alpha=0.2)
    axes[2].legend(fontsize=8)
    fig.suptitle("二维二连杆机械臂：关节速度 CBF 迁移原型", fontsize=14)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "nominal_min_margin": float(nominal.minimum_margin.min()),
        "filtered_min_margin": float(safe.minimum_margin.min()),
        "nominal_reached_goal": nominal.reached_goal,
        "filtered_reached_goal": safe.reached_goal,
        "filter_intervention_rate": float(safe.filter_active.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("logs/plots/arm2d_cbf_demo.png"))
    args = parser.parse_args()
    metrics = make_figure(args.output)
    print(f"saved: {args.output.resolve()}")
    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
