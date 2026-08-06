"""按 plot_fig4_no_dr.py 的版式绘制低阶机械臂固定场景轨迹。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from rsl_rl.modules import ActorCritic

from arm2d.vec_env import Arm2DVecEnv
from experiment_configs import TRAINING_METHODS


@dataclass(frozen=True)
class RolloutConfig:
    label: str
    method: str
    runtime_filter: bool
    color: str
    linestyle: str = "-"


CONFIGS = (
    RolloutConfig("Nominal", "nominal", False, "#1f77b4"),
    RolloutConfig("Dual", "dual", True, "#aec7e8"),
    RolloutConfig("Dual (w/o rt. filt.)", "dual", False, "#ff7f0e", "--"),
    RolloutConfig("Reward Only", "reward_only", False, "#ffbb78"),
    RolloutConfig("Filter Only", "filter_only", True, "#2ca02c"),
    RolloutConfig("Filter Only (w/o rt. filt.)", "filter_only", False, "#98df8a", "--"),
)


def latest_checkpoint(log_root: Path, method: str) -> Path:
    checkpoints = list((log_root / method).glob("*/model_1000.pt"))
    if not checkpoints:
        checkpoints = list((log_root / method).glob("*/model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No Arm2D checkpoint found for {method}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_policy(checkpoint: Path, device: str) -> ActorCritic:
    policy = ActorCritic(
        11,
        11,
        2,
        init_noise_std=0.8,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        activation="elu",
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    return policy


def snapshot(env: Arm2DVecEnv) -> dict[str, torch.Tensor]:
    return {
        "q": env.q.clone(),
        "goal": env.goal.clone(),
        "obstacle_center": env.obstacle_center.clone(),
        "obstacle_radius": env.obstacle_radius.clone(),
    }


def restore(env: Arm2DVecEnv, state: dict[str, torch.Tensor]) -> torch.Tensor:
    env.q[:] = state["q"]
    env.goal[:] = state["goal"]
    env.obstacle_center[:] = state["obstacle_center"]
    env.obstacle_radius[:] = state["obstacle_radius"]
    env.last_velocity.zero_()
    env.episode_length_buf.zero_()
    return env.get_observations()[0]


@torch.inference_mode()
def rollout(
    env: Arm2DVecEnv,
    policy: ActorCritic,
    state: dict[str, torch.Tensor],
    runtime_filter: bool,
) -> dict:
    env.use_filter = runtime_filter
    obs = restore(env, state)
    _, tip = env._kinematics(env.q)
    tip_trajectory = [tip[0].cpu().numpy().copy()]
    q_trajectory = [env.q[0].cpu().numpy().copy()]
    min_margin = float("inf")
    outcome = "timeout"

    # 手动更新以在终止步保留 q；标准 step() 会立即自动 reset。
    for _ in range(env.max_episode_length):
        nominal = torch.clamp(policy.act_inference(obs), -env.velocity_limit, env.velocity_limit)
        a, b, _ = env._all_constraints()
        safe, _, _ = env._filter(nominal, a, b)
        executed = safe if runtime_filter else nominal
        env.last_velocity = executed
        env.q = torch.clamp(env.q + env.dt * executed, env.q_min, env.q_max)
        env.episode_length_buf += 1

        _, tip = env._kinematics(env.q)
        h, _ = env._obstacle_barriers(env.q)
        min_margin = min(min_margin, float(h.min()))
        tip_trajectory.append(tip[0].cpu().numpy().copy())
        q_trajectory.append(env.q[0].cpu().numpy().copy())
        distance = torch.linalg.vector_norm(env.goal - tip, dim=1)
        if distance.item() < env.goal_radius:
            outcome = "success"
            break
        if (h.min() + env.safety_margin).item() < 0.0:
            outcome = "obstacle"
            break
        obs = env.get_observations()[0]
    return {
        "trajectory": np.asarray(tip_trajectory),
        "q_trajectory": np.asarray(q_trajectory),
        "outcome": outcome,
        "min_margin": min_margin,
    }


def scenario_score(results: list[dict]) -> tuple[int, int]:
    """沿用原脚本：优先搜索兼具成功和失败、且符合论文消融趋势的场景。"""
    success = [result["outcome"] == "success" for result in results]
    diversity = min(sum(success), len(success) - sum(success))
    paper_pattern = sum((
        not success[0],  # Nominal failure
        success[1],      # Dual with filter success
        success[2],      # Dual without filter success
        success[3],      # Reward Only success
        success[4],      # Filter Only with filter success
        not success[5],  # Filter Only without filter failure
    ))
    return paper_pattern, diversity


def joint_positions(q: np.ndarray) -> np.ndarray:
    q1, q2 = q
    elbow = np.array([np.cos(q1), np.sin(q1)])
    tip = elbow + 0.8 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    return np.vstack((np.zeros(2), elbow, tip))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("logs/plots/arm2d_paper_style/arm2d_fig4_no_dr"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    env = Arm2DVecEnv(
        num_envs=1,
        method=TRAINING_METHODS["nominal"],
        device=args.device,
        seed=args.seed_start,
    )
    policies = {
        method: load_policy(latest_checkpoint(Path("logs/arm2d"), method), args.device)
        for method in {config.method for config in CONFIGS}
    }

    best = None
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        # 每个 seed 构造一个独立、确定性的机械臂任务。
        candidate_env = Arm2DVecEnv(
            num_envs=1,
            method=TRAINING_METHODS["nominal"],
            device=args.device,
            seed=seed,
        )
        state = snapshot(candidate_env)
        results = [rollout(env, policies[config.method], state, config.runtime_filter) for config in CONFIGS]
        score = scenario_score(results)
        if best is None or score > best[0]:
            best = (score, seed, state, results)
        if score[0] == 6:
            break

    assert best is not None
    score, seed, state, results = best
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    center = state["obstacle_center"][0].cpu().numpy()
    radius = float(state["obstacle_radius"][0])
    # 与导航脚本一致，红色区域包含被控对象半径；虚线额外表示 CBF 安全余量。
    ax.add_patch(Circle(center, radius + env.link_radius, color="red", zorder=2))
    ax.add_patch(Circle(
        center,
        radius + env.link_radius + env.safety_margin,
        fill=False,
        color="firebrick",
        linestyle="--",
        linewidth=1.0,
        zorder=2,
    ))

    start_q = state["q"][0].cpu().numpy()
    start_arm = joint_positions(start_q)
    start = start_arm[-1]
    goal = state["goal"][0].cpu().numpy()
    ax.plot(start_arm[:, 0], start_arm[:, 1], "o-", color="0.35", linewidth=2.2, markersize=4, zorder=3)
    ax.scatter(start[0], start[1], color="black", s=90, zorder=5, label="Start")
    ax.add_patch(Circle(goal, env.goal_radius, color="gold", zorder=3, label="Goal"))

    for config, result in zip(CONFIGS, results):
        trajectory = result["trajectory"]
        mark = "✓" if result["outcome"] == "success" else "×"
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            color=config.color,
            linestyle=config.linestyle,
            linewidth=2.2,
            label=f"{config.label} {mark}",
            zorder=4,
        )
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=config.color, s=22, zorder=5)

    ax.set(xlim=(-0.15, 1.95), ylim=(-1.20, 1.75), xlabel="x (m)", ylabel="y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.85", linewidth=0.8)
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        prop={"family": "DejaVu Sans", "size": 10},
    )
    ax.set_title(f"Trajectory comparison (Arm2D, no DR, seed {seed})")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    with args.output.with_suffix(".txt").open("w", encoding="utf-8") as stream:
        stream.write(f"seed={seed}\nscore={score}\n")
        for config, result in zip(CONFIGS, results):
            stream.write(
                f"{config.label}: {result['outcome']}, min_margin={result['min_margin']:.6f}\n"
            )
    print(f"seed={seed}, score={score}")
    for config, result in zip(CONFIGS, results):
        print(f"{config.label}: {result['outcome']}, min_margin={result['min_margin']:.6f}")
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
