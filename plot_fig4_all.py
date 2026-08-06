"""Reproduce paper Fig. 4 with all 12 deployment configurations."""

from __future__ import annotations

import argparse
import csv
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
class RunConfig:
    label: str
    method: str
    runtime_filter: bool
    dr: bool
    color: str
    linestyle: str = "-"


RUNS = (
    RunConfig("Nominal", "nominal", False, False, "#1f77b4"),
    RunConfig("Dual", "dual", True, False, "#aec7e8"),
    RunConfig("Dual (w/o rt. filt.)", "dual", False, False, "#ff7f0e", "--"),
    RunConfig("Reward Only", "reward_only", False, False, "#ffbb78"),
    RunConfig("Filter Only", "filter_only", True, False, "#2ca02c"),
    RunConfig("Filter Only (w/o rt. filt.)", "filter_only", False, False, "#98df8a", "--"),
    RunConfig("Nominal DR", "nominal", False, True, "#d62728"),
    RunConfig("Dual DR", "dual", True, True, "#ff9896"),
    RunConfig("Dual (w/o rt. filt.) DR", "dual", False, True, "#9467bd", "--"),
    RunConfig("Reward Only DR", "reward_only", False, True, "#c5b0d5"),
    RunConfig("Filter Only DR", "filter_only", True, True, "#8c564b"),
    RunConfig("Filter Only (w/o rt. filt.) DR", "filter_only", False, True, "#c49c94", "--"),
)


def latest_checkpoint(log_root: Path, method: str, dr: bool) -> Path:
    experiment = f"navigation_{method}{'_dr' if dr else ''}"
    checkpoints = list((log_root / experiment).rglob("model_1499.pt"))
    if not checkpoints:
        checkpoints = list((log_root / experiment).rglob("model_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint under {log_root / experiment}")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_policy(path: Path, obs_dim: int, action_dim: int) -> ActorCritic:
    kwargs = dict(cfg["policy"])
    kwargs.pop("class_name", None)
    policy = ActorCritic(obs_dim, obs_dim, action_dim, **kwargs).cpu()
    policy.load_state_dict(torch.load(path, map_location="cpu")["model_state_dict"])
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
    run: RunConfig,
    scenario_seed: int,
) -> dict:
    env.use_cbf_action_filtering = run.runtime_filter
    env.noise_level = 0.2 if run.dr else 0.0
    # Every DR configuration receives the same disturbance sequence for a
    # given scene. This makes trajectory differences attributable to policy.
    torch.manual_seed(100_000 + scenario_seed)
    obs = restore(env, state)
    trajectory = [state["robot"][0].numpy().copy()]
    outcome = "timeout"
    steps = env.max_episode_length

    for step in range(env.max_episode_length):
        with torch.inference_mode():
            action = policy.act_inference(obs)
        obs, _, done, extras = env.step(action)
        trajectory.append(extras["robot_position"][0].numpy().copy())
        if done.item():
            steps = step + 1
            if extras["log"]["success"].item():
                outcome = "success"
            elif extras["log"]["collided_obstacle"].item():
                outcome = "obstacle"
            elif extras["log"]["collided_wall"].item():
                outcome = "wall"
            break
    return {"trajectory": np.asarray(trajectory), "outcome": outcome, "steps": steps}


def draw_scene(
    env: UnifiedNavigationEnv,
    state: dict[str, torch.Tensor],
    results: list[dict],
    seed: int,
    num_obstacles: int,
    output: Path,
) -> None:
    plt.rcParams["font.family"] = "Times New Roman"
    fig, ax = plt.subplots(figsize=(8.4, 6.4))

    for center, radius in zip(state["obstacles"][0].numpy(), state["radii"][0].numpy()):
        ax.add_patch(Circle(center, radius + env.robot_radius, color="red", zorder=2))
    start = state["robot"][0].numpy()
    goal = state["goal"][0].numpy()
    ax.scatter(*start, color="black", s=85, zorder=6, label="Start")
    ax.add_patch(Circle(goal, env.goal_radius + env.robot_radius, color="gold", zorder=3, label="Goal"))

    for run, result in zip(RUNS, results):
        xy = result["trajectory"]
        mark = "✓" if result["outcome"] == "success" else "×"
        ax.plot(
            xy[:, 0], xy[:, 1], color=run.color, linestyle=run.linestyle,
            linewidth=1.9, label=f"{run.label} {mark}", zorder=4,
        )
        ax.scatter(*xy[-1], color=run.color, s=18, zorder=5)

    ax.set(xlim=(0, env.world_size), ylim=(0, env.world_size), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.86", linewidth=0.8)
    ax.set_title(f"Trajectory comparison ({num_obstacles} obstacles, seed {seed})")
    ax.legend(
        loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
        prop={"family": "DejaVu Sans", "size": 8.5},
    )
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--num-obstacles", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("logs/plots/fig4_all"))
    args = parser.parse_args()

    env_kwargs = {key: value for key, value in cfg["env"].items() if key not in {"env_id", "num_envs", "noise_level"}}
    env_kwargs["num_obstacles"] = args.num_obstacles
    env = UnifiedNavigationEnv(
        num_envs=1, device="cpu", noise_level=0.0,
        use_cbf_reward_penalty=False, **env_kwargs,
    )
    obs, _ = env.get_observations()
    policies = {
        (run.method, run.dr): load_policy(
            latest_checkpoint(Path("logs"), run.method, run.dr), obs.shape[1], env.num_actions
        )
        for run in RUNS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        env.reset(seed=seed)
        state = snapshot(env)
        results = [rollout(env, policies[(run.method, run.dr)], state, run, seed) for run in RUNS]
        output = args.output_dir / f"fig4_5obs_seed_{seed}"
        draw_scene(env, state, results, seed, args.num_obstacles, output)
        for run, result in zip(RUNS, results):
            rows.append({
                "seed": seed,
                "configuration": run.label,
                "training_method": run.method,
                "runtime_filter": run.runtime_filter,
                "dr": run.dr,
                "outcome": result["outcome"],
                "steps": result["steps"],
            })
        print(f"seed={seed}: " + ", ".join(f"{run.label}={result['outcome']}" for run, result in zip(RUNS, results)))

    with (args.output_dir / "fig4_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    env.close()


if __name__ == "__main__":
    main()
