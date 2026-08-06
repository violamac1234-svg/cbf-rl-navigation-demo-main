"""按 plot_paper_fig3.py 的版式绘制低阶机械臂训练曲线。"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METHODS = {
    "Nominal": ("nominal", "#d62728"),
    "Reward Only": ("reward_only", "#ff7f0e"),
    "Filter Only": ("filter_only", "#2ca02c"),
    "Dual": ("dual", "#1f77b4"),
}


def latest_completed_event(log_root: Path, method: str) -> Path:
    """只选择带 model_1000.pt 的正式训练，避免误读 5 次迭代的冒烟日志。"""
    candidates = []
    for run in (log_root / method).iterdir():
        events = list(run.glob("events.out.tfevents*"))
        if (run / "model_1000.pt").exists() and events:
            candidates.extend(events)
    if not candidates:
        raise FileNotFoundError(f"No completed Arm2D TensorBoard run under {log_root / method}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def scalars(event_path: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    values = accumulator.Scalars(tag)
    if not values:
        raise KeyError(f"TensorBoard tag {tag!r} is missing in {event_path}")
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
    """局部 5%/95% 分位包络，仅表示训练波动，不是置信区间。"""
    radius = window // 2
    low = np.empty_like(values, dtype=float)
    high = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        local = values[max(0, index - radius): min(len(values), index + radius + 1)]
        low[index], high[index] = np.percentile(local, [5, 95])
    return low, high


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/arm2d"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs/plots/arm2d_paper_style"))
    parser.add_argument("--smooth", type=float, default=0.92)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--reward-ymin", type=float, default=-65.0)
    parser.add_argument("--reward-ymax", type=float, default=90.0)
    parser.add_argument("--collision-ymax", type=float, default=65.0)
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

    for label, (method, color) in METHODS.items():
        event = latest_completed_event(args.log_root, method)
        reward_steps, rewards = scalars(event, "Train/mean_reward")
        collision_steps, collision_mean = scalars(event, "arm/collision")

        reward_low, reward_high = rolling_envelope(rewards)
        reward_ax.fill_between(
            reward_steps, reward_low, reward_high, color=color, alpha=0.12, linewidth=0
        )
        reward_ax.plot(
            reward_steps, ema(rewards, args.smooth), color=color, linewidth=2.0, label=label
        )

        # 与原 plot_paper_fig3.py 一致：把并行环境的平均碰撞指示量换算为事件数尺度。
        collision_count = collision_mean * args.num_envs
        collision_ax.plot(
            collision_steps,
            ema(collision_count, args.smooth),
            color=color,
            linewidth=2.0,
            label=label,
        )

    reward_ax.set_xlabel("Training iteration")
    reward_ax.set_ylabel("Mean episode reward")
    reward_ax.set_xlim(0, 1000)
    reward_ax.set_ylim(args.reward_ymin, args.reward_ymax)
    reward_ax.legend(loc="lower right", ncol=2, frameon=True)

    collision_ax.set_xlabel("Training iteration")
    collision_ax.set_ylabel(f"Collisions per {args.num_envs} environments")
    collision_ax.set_xlim(0, 1000)
    collision_ax.set_ylim(0, args.collision_ymax)
    collision_ax.legend(loc="upper right", ncol=2, frameon=True)

    for axis in (reward_ax, collision_ax):
        axis.grid(True, color="0.88", linewidth=0.7)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    reward_fig.tight_layout()
    collision_fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = (
        (reward_fig, args.output_dir / "arm2d_paper_reward_curve"),
        (collision_fig, args.output_dir / "arm2d_paper_collision_curve"),
    )
    for figure, output in outputs:
        figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
        figure.savefig(output.with_suffix(".png"), dpi=260, bbox_inches="tight")
        plt.close(figure)
        print(output.with_suffix(".pdf"))
        print(output.with_suffix(".png"))


if __name__ == "__main__":
    main()
