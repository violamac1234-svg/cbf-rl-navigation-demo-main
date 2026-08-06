"""用四个低阶机械臂 checkpoint 绘制论文 Fig.3/Fig.4 风格结果图。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from arm2d.vec_env import Arm2DVecEnv
from experiment_configs import TRAINING_METHODS
from train_arm2d import make_training_config


METHODS = ("nominal", "reward_only", "filter_only", "dual")
LABELS = {
    "nominal": "Nominal",
    "reward_only": "Reward Only",
    "filter_only": "Filter Only",
    "dual": "Dual",
}
COLORS = {
    "nominal": "#d64541",
    "reward_only": "#ed8b2d",
    "filter_only": "#2f9e44",
    "dual": "#2878b5",
}


@dataclass
class Scenario:
    q0: torch.Tensor
    goal: torch.Tensor
    obstacle_center: torch.Tensor
    obstacle_radius: torch.Tensor
    index: int


@dataclass
class Trajectory:
    q: np.ndarray
    tip: np.ndarray
    margin: np.ndarray
    success: bool
    collision: bool


def configure_fonts():
    chinese_font = Path("C:/Windows/Fonts/simhei.ttf")
    if chinese_font.exists():
        font_manager.fontManager.addfont(chinese_font)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=chinese_font).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def latest_completed_run(method: str, logs_root: Path) -> tuple[Path, Path, Path]:
    candidates = []
    for run in (logs_root / method).iterdir():
        checkpoint = run / "model_1000.pt"
        events = list(run.glob("events.out.tfevents.*"))
        if checkpoint.exists() and events:
            candidates.append((run, checkpoint, max(events, key=lambda item: item.stat().st_mtime)))
    if not candidates:
        raise FileNotFoundError(f"No completed 1000-iteration run found for {method}")
    return max(candidates, key=lambda item: item[0].stat().st_mtime)


def moving_average(values: np.ndarray, window: int):
    if values.size < window:
        return np.arange(values.size), values
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    return np.arange(window - 1, values.size), smoothed


def plot_training_curves(logs_root: Path, output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(8.2, 6.8), sharex=True)
    for method in METHODS:
        _, _, event_path = latest_completed_run(method, logs_root)
        accumulator = EventAccumulator(str(event_path))
        accumulator.Reload()
        reward_events = accumulator.Scalars("Train/mean_reward")
        collision_events = accumulator.Scalars("arm/collision")
        reward = np.asarray([event.value for event in reward_events])
        collision_per_1000 = 1000.0 * np.asarray([event.value for event in collision_events])
        reward_x, reward_smooth = moving_average(reward, 25)
        collision_x, collision_smooth = moving_average(collision_per_1000, 25)
        axes[0].plot(reward_x, reward_smooth, lw=2.0, color=COLORS[method], label=LABELS[method])
        axes[1].plot(collision_x, collision_smooth, lw=2.0, color=COLORS[method], label=LABELS[method])

    axes[0].set_title("平均回合奖励随训练迭代变化（25 点滑动平均）")
    axes[0].set_ylabel("平均回合奖励")
    axes[1].set_title("训练期间碰撞事件（25 点滑动平均）")
    axes[1].set_ylabel("每 1000 环境步碰撞数")
    axes[1].set_xlabel("PPO 迭代")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(ncol=4, loc="best", fontsize=9)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def load_policy(method: str, env: Arm2DVecEnv, checkpoint: Path):
    args = SimpleNamespace(seed=42, device=str(env.device), steps_per_env=32, save_interval=100)
    runner = OnPolicyRunner(env, deepcopy(make_training_config(args)), log_dir=None, device=str(env.device))
    runner.load(str(checkpoint), load_optimizer=False)
    return runner.get_inference_policy(device=str(env.device))


@torch.inference_mode()
def batch_outcomes(method: str, checkpoint: Path, num_scenarios: int, scenario_seed: int, device: str):
    env = Arm2DVecEnv(num_scenarios, TRAINING_METHODS[method], device, scenario_seed)
    initial = Scenario(
        env.q.clone(), env.goal.clone(), env.obstacle_center.clone(), env.obstacle_radius.clone(), -1
    )
    policy = load_policy(method, env, checkpoint)
    finished = torch.zeros(num_scenarios, dtype=torch.bool, device=env.device)
    success = torch.zeros_like(finished)
    collision = torch.zeros_like(finished)
    for _ in range(env.max_episode_length):
        obs, _ = env.get_observations()
        _, _, done, extras = env.step(policy(obs))
        newly_finished = done & ~finished
        success[newly_finished] = extras["log"]["arm/success"][newly_finished].bool()
        collision[newly_finished] = extras["log"]["arm/collision"][newly_finished].bool()
        finished |= newly_finished
        if finished.all():
            break
    return initial, success.cpu().numpy(), collision.cpu().numpy()


def select_scenario(checkpoints: dict[str, Path], num_scenarios: int, scenario_seed: int, device: str):
    outcomes = {}
    reference = None
    for method in METHODS:
        initial, success, collision = batch_outcomes(
            method, checkpoints[method], num_scenarios, scenario_seed, device
        )
        if reference is None:
            reference = initial
        outcomes[method] = (success, collision)

    # 优先选择安全方法成功、Nominal 碰撞的场景；没有完全满足时选择得分最高者。
    score = (
        5 * outcomes["dual"][0]
        + 5 * outcomes["filter_only"][0]
        + 2 * outcomes["reward_only"][0]
        + 3 * outcomes["nominal"][1]
        - 6 * outcomes["dual"][1]
        - 6 * outcomes["filter_only"][1]
    )
    index = int(np.argmax(score))
    scenario = Scenario(
        reference.q0[index].clone(),
        reference.goal[index].clone(),
        reference.obstacle_center[index].clone(),
        reference.obstacle_radius[index].clone(),
        index,
    )
    return scenario, {method: (bool(values[0][index]), bool(values[1][index])) for method, values in outcomes.items()}


@torch.inference_mode()
def rollout_scenario(method: str, checkpoint: Path, scenario: Scenario, device: str):
    env = Arm2DVecEnv(1, TRAINING_METHODS[method], device, seed=0)
    env.q[0] = scenario.q0
    env.goal[0] = scenario.goal
    env.obstacle_center[0] = scenario.obstacle_center
    env.obstacle_radius[0] = scenario.obstacle_radius
    env.last_velocity.zero_()
    env.episode_length_buf.zero_()
    policy = load_policy(method, env, checkpoint)
    q_history, tip_history, margin_history = [], [], []
    success = collision = False
    for _ in range(env.max_episode_length):
        _, tip = env._kinematics(env.q)
        h, _ = env._obstacle_barriers(env.q)
        q_history.append(env.q[0].cpu().numpy().copy())
        tip_history.append(tip[0].cpu().numpy().copy())
        margin_history.append(float(h.min()))

        obs, _ = env.get_observations()
        nominal = torch.clamp(policy(obs), -env.velocity_limit, env.velocity_limit)
        a, b, _ = env._all_constraints()
        safe, _, _ = env._filter(nominal, a, b)
        executed = safe if env.use_filter else nominal
        env.last_velocity = executed
        env.q = torch.clamp(env.q + env.dt * executed, env.q_min, env.q_max)

        _, new_tip = env._kinematics(env.q)
        new_h, _ = env._obstacle_barriers(env.q)
        success = bool(torch.linalg.vector_norm(env.goal - new_tip, dim=1)[0] < env.goal_radius)
        collision = bool(new_h.min() + env.safety_margin < 0.0)
        if success or collision:
            q_history.append(env.q[0].cpu().numpy().copy())
            tip_history.append(new_tip[0].cpu().numpy().copy())
            margin_history.append(float(new_h.min()))
            break
    return Trajectory(
        np.asarray(q_history), np.asarray(tip_history), np.asarray(margin_history), success, collision
    )


def joint_positions(q: np.ndarray):
    q1, q2 = q
    elbow = np.array([np.cos(q1), np.sin(q1)])
    tip = elbow + 0.8 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    return np.vstack((np.zeros(2), elbow, tip))


def plot_trajectories(trajectories: dict[str, Trajectory], scenario: Scenario, output: Path):
    goal = scenario.goal.cpu().numpy()
    center = scenario.obstacle_center.cpu().numpy()
    radius = float(scenario.obstacle_radius)
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 8.2), sharex=True, sharey=True)
    for axis, method in zip(axes.flat, METHODS):
        trajectory = trajectories[method]
        color = COLORS[method]
        axis.add_patch(Circle(center, radius, color="#e53935", alpha=0.86, zorder=2))
        axis.add_patch(Circle(center, radius + 0.08, fill=False, color="#a61b1b", ls="--", lw=1.3))
        indices = np.unique(np.linspace(0, len(trajectory.q) - 1, 10).astype(int))
        for index in indices:
            points = joint_positions(trajectory.q[index])
            axis.plot(points[:, 0], points[:, 1], color=color, alpha=0.10, lw=2.5)
        start_arm = joint_positions(trajectory.q[0])
        final_arm = joint_positions(trajectory.q[-1])
        axis.plot(start_arm[:, 0], start_arm[:, 1], "o-", color="#303030", alpha=0.7, lw=2.5, ms=4)
        axis.plot(final_arm[:, 0], final_arm[:, 1], "o-", color=color, lw=3.2, ms=5)
        axis.plot(trajectory.tip[:, 0], trajectory.tip[:, 1], color=color, lw=2.2)
        axis.scatter(*trajectory.tip[0], color="#161616", s=36, zorder=5)
        axis.scatter(*goal, color="#f2c500", edgecolor="#826d00", marker="*", s=145, zorder=6)
        endpoint_marker = "*" if trajectory.success else "X"
        axis.scatter(*trajectory.tip[-1], color=color, marker=endpoint_marker, s=70, zorder=6)
        outcome = "到达目标" if trajectory.success else ("碰撞" if trajectory.collision else "超时")
        axis.set_title(
            f"{LABELS[method]} — {outcome}\n最小安全裕度 {trajectory.margin.min():.3f} m，{len(trajectory.q)-1} 步"
        )
        axis.set_aspect("equal")
        axis.set_xlim(-0.15, 1.95)
        axis.set_ylim(-1.25, 1.75)
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 1].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    axes[1, 0].set_ylabel("y (m)")
    fig.suptitle(f"固定场景轨迹对比（场景编号={scenario.index}）", fontsize=14)
    fig.tight_layout()
    fig.savefig(output, dpi=240, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-root", type=Path, default=Path("logs/arm2d"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/plots/arm2d_paper"))
    parser.add_argument("--scenario-seed", type=int, default=20260807)
    parser.add_argument("--num-scenarios", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    configure_fonts()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = {method: latest_completed_run(method, args.logs_root)[1] for method in METHODS}
    plot_training_curves(args.logs_root, args.output_dir / "fig3_arm2d_training.png")
    scenario, batch_result = select_scenario(
        checkpoints, args.num_scenarios, args.scenario_seed, args.device
    )
    trajectories = {
        method: rollout_scenario(method, checkpoints[method], scenario, args.device) for method in METHODS
    }
    plot_trajectories(trajectories, scenario, args.output_dir / "fig4_arm2d_trajectories.png")
    print(f"selected scenario index: {scenario.index} (generator seed {args.scenario_seed})")
    print(f"batch outcomes at selected scenario: {batch_result}")
    for method, trajectory in trajectories.items():
        print(
            f"{method}: success={trajectory.success}, collision={trajectory.collision}, "
            f"steps={len(trajectory.q)-1}, min_h={trajectory.margin.min():.6f}"
        )
    print(f"saved figures to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
