"""Create separate paper-style reward and collision figures for four methods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METHODS = {
    "Nominal": ("navigation_nominal", "#d62728"),
    "Reward Only": ("navigation_reward_only", "#ff7f0e"),
    "Filter Only": ("navigation_filter_only", "#2ca02c"),
    "Dual": ("navigation_dual", "#1f77b4"),
}


def latest_event(log_root: Path, experiment: str) -> Path:
    events = list((log_root / experiment).rglob("events.out.tfevents*"))
    if not events:
        raise FileNotFoundError(f"No TensorBoard event file under {log_root / experiment}")
    return max(events, key=lambda path: path.stat().st_mtime)


def scalars(event_path: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    return (
        np.asarray([item.step for item in values], dtype=float),
        np.asarray([item.value for item in values], dtype=float),
    )


def ema(values: np.ndarray, smoothing: float) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    result[0] = values[0]
    for index in range(1, len(values)):
        result[index] = smoothing * result[index - 1] + (1.0 - smoothing) * values[index]
    return result


def rolling_envelope(values: np.ndarray, window: int = 21) -> tuple[np.ndarray, np.ndarray]:
    """Local 5th/95th-percentile envelope (visual variability, not a CI)."""
    radius = window // 2
    low = np.empty_like(values, dtype=float)
    high = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        local = values[max(0, index - radius): min(len(values), index + radius + 1)]
        low[index], high[index] = np.percentile(local, [5, 95])
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/plots"))
    parser.add_argument("--smooth", type=float, default=0.92)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    reward_fig, reward_ax = plt.subplots(figsize=(7.2, 4.4))
    collision_fig, collision_ax = plt.subplots(figsize=(7.2, 4.4))

    for method, (directory, color) in METHODS.items():
        event = latest_event(args.log_root, directory)
        # Fig. 3 uses rsl_rl's accumulated episode return, not Episode/reward_log.
        reward_steps, rewards = scalars(event, "Train/mean_reward")
        obstacle_steps, obstacle = scalars(event, "Episode/collided_obstacle")
        wall_steps, wall = scalars(event, "Episode/collided_wall")
        if not np.array_equal(obstacle_steps, wall_steps):
            raise ValueError(f"Collision series are not aligned for {method}")

        reward_low, reward_high = rolling_envelope(rewards)
        reward_ax.fill_between(reward_steps, reward_low, reward_high, color=color, alpha=0.12, linewidth=0)
        reward_ax.plot(reward_steps, ema(rewards, args.smooth), color=color, linewidth=2.0, label=method)

        # TensorBoard stores a mean collision indicator. Multiplication by the
        # 4096 parallel environments reproduces the collision-count scale.
        collision_count = (obstacle + wall) * 4096.0
        collision_ax.plot(
            obstacle_steps,
            ema(collision_count, args.smooth),
            color=color,
            linewidth=2.0,
            label=method,
        )

    reward_ax.set_xlabel("Training iteration")
    reward_ax.set_ylabel("Mean episode reward")
    reward_ax.set_xlim(0, 1500)
    reward_ax.set_ylim(0, 2550)
    reward_ax.legend(loc="lower right", ncol=2, frameon=True)

    collision_ax.set_xlabel("Training iteration")
    collision_ax.set_ylabel("Collisions per 4096 environments")
    collision_ax.set_xlim(0, 1500)
    collision_ax.set_ylim(0, 38)
    collision_ax.legend(loc="upper right", ncol=2, frameon=True)

    for axis in (reward_ax, collision_ax):
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    reward_fig.tight_layout()
    collision_fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (reward_fig, args.output_dir / "paper_reward_curve"),
        (collision_fig, args.output_dir / "paper_collision_curve"),
    )
    for figure, output in outputs:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
        print(output.with_suffix(".pdf"))
        print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
