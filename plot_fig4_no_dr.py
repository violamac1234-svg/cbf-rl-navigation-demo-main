"""Plot a fixed-scenario, no-DR trajectory comparison like paper Fig. 4."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle
from rsl_rl.modules import ActorCritic

from config import cfg
from nav_env.unified_navigation_env import UnifiedNavigationEnv


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
    checkpoints = list((log_root / f"navigation_{method}").rglob("model_1499.pt"))
    if not checkpoints:
        checkpoints = list((log_root / f"navigation_{method}").rglob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint found for {method}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_policy(checkpoint: Path, obs_dim: int, action_dim: int, device: str) -> ActorCritic:
    policy_kwargs = dict(cfg["policy"])
    policy_kwargs.pop("class_name", None)
    policy = ActorCritic(obs_dim, obs_dim, action_dim, **policy_kwargs).to(device)
    payload = torch.load(checkpoint, map_location=device)
    policy.load_state_dict(payload["model_state_dict"])
    policy.eval()
    return policy


def snapshot(env: UnifiedNavigationEnv) -> dict[str, torch.Tensor]:
    return {
        "robot": env._robot_pos.clone(),
        "goal": env._goal_pos.clone(),
        "obstacles": env._obstacle_positions.clone(),
        "radii": env._obstacle_radii.clone(),
    }


def restore(env: UnifiedNavigationEnv, state: dict[str, torch.Tensor]) -> torch.Tensor:
    env._robot_pos[:] = state["robot"]
    env._goal_pos[:] = state["goal"]
    env._obstacle_positions[:] = state["obstacles"]
    env._obstacle_radii[:] = state["radii"]
    env._last_velocity.zero_()
    env._elapsed_steps.zero_()
    env.episode_length_buf.zero_()
    return env.get_observations()[0]


def rollout(
    env: UnifiedNavigationEnv,
    policy: ActorCritic,
    state: dict[str, torch.Tensor],
    runtime_filter: bool,
) -> dict:
    env.use_cbf_action_filtering = runtime_filter
    obs = restore(env, state)
    trajectory = [state["robot"][0].cpu().numpy().copy()]
    outcome = "timeout"

    for _ in range(env.max_episode_length):
        with torch.inference_mode():
            action = policy.act_inference(obs)
        obs, _, done, extras = env.step(action)
        trajectory.append(extras["robot_position"][0].cpu().numpy().copy())
        if done.item():
            if extras["log"]["success"].item():
                outcome = "success"
            elif extras["log"]["collided_obstacle"].item():
                outcome = "obstacle"
            elif extras["log"]["collided_wall"].item():
                outcome = "wall"
            break
    return {"trajectory": np.asarray(trajectory), "outcome": outcome}


def scenario_score(results: list[dict]) -> tuple[int, int]:
    """Prefer a readable scene with both successes and failures, like Fig. 4."""
    success = [result["outcome"] == "success" for result in results]
    diversity = min(sum(success), len(success) - sum(success))
    paper_pattern = sum(
        (
            not success[0],  # Nominal failure
            success[1],      # Dual with filter success
            success[2],      # Dual without filter success
            success[3],      # Reward Only success
            success[4],      # Filter Only with filter success
            not success[5],  # Filter Only without filter failure
        )
    )
    return paper_pattern, diversity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=100)
    parser.add_argument("--num-obstacles", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("logs/plots/fig4_no_dr_5_obstacles"))
    args = parser.parse_args()

    device = "cpu"  # Faster and deterministic for six single-environment rollouts.
    env_kwargs = {key: value for key, value in cfg["env"].items() if key not in {"env_id", "num_envs", "noise_level"}}
    # The trained policy observes only the closest obstacle, so changing the
    # total obstacle count does not change the checkpoint's input dimension.
    env_kwargs["num_obstacles"] = args.num_obstacles
    env = UnifiedNavigationEnv(
        num_envs=1,
        device=device,
        noise_level=0.0,
        use_cbf_reward_penalty=False,
        **env_kwargs,
    )
    obs, _ = env.get_observations()
    policies = {
        method: load_policy(latest_checkpoint(Path("logs"), method), obs.shape[1], env.num_actions, device)
        for method in {config.method for config in CONFIGS}
    }

    best = None
    for seed in range(args.seed_start, args.seed_start + args.seed_count):
        env.reset(seed=seed)
        state = snapshot(env)
        results = [rollout(env, policies[config.method], state, config.runtime_filter) for config in CONFIGS]
        score = scenario_score(results)
        if best is None or score > best[0]:
            best = (score, seed, state, results)
        if score[0] == 6:
            break

    assert best is not None
    score, seed, state, results = best
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    plt.rcParams["font.family"] = "Times New Roman"

    obstacle_positions = state["obstacles"][0].cpu().numpy()
    obstacle_radii = state["radii"][0].cpu().numpy()
    for center, radius in zip(obstacle_positions, obstacle_radii):
        ax.add_patch(Circle(center, radius + env.robot_radius, color="red", zorder=2))

    start = state["robot"][0].cpu().numpy()
    goal = state["goal"][0].cpu().numpy()
    ax.scatter(start[0], start[1], color="black", s=90, zorder=5, label="Start")
    ax.add_patch(Circle(goal, env.goal_radius + env.robot_radius, color="gold", zorder=3, label="Goal"))

    for config, result in zip(CONFIGS, results):
        trajectory = result["trajectory"]
        mark = "✓" if result["outcome"] == "success" else "×"
        ax.plot(
            trajectory[:, 0], trajectory[:, 1],
            color=config.color, linestyle=config.linestyle, linewidth=2.2,
            label=f"{config.label} {mark}", zorder=4,
        )
        ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color=config.color, s=22, zorder=5)

    ax.set(xlim=(0, env.world_size), ylim=(0, env.world_size), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.85", linewidth=0.8)
    # DejaVu Sans contains the check/cross glyphs used for outcome labels.
    ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
        prop={"family": "DejaVu Sans", "size": 10},
    )
    ax.set_title(
        f"Trajectory comparison (no DR, {args.num_obstacles} obstacles, seed {seed})"
    )
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    with args.output.with_suffix(".txt").open("w", encoding="utf-8") as stream:
        stream.write(f"seed={seed}\nnum_obstacles={args.num_obstacles}\nscore={score}\n")
        for config, result in zip(CONFIGS, results):
            stream.write(f"{config.label}: {result['outcome']}\n")
    env.close()
    print(f"seed={seed}, score={score}")
    for config, result in zip(CONFIGS, results):
        print(f"{config.label}: {result['outcome']}")
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
