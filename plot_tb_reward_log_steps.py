#!/usr/bin/env python3
"""
Plot "Mean episode reward_log" from TensorBoard event files vs training steps.

本文件只负责读取 TensorBoard 标量并重绘论文风格曲线，不参与环境动力学、
CBF 过滤或 PPO 更新。安全项的原始数据由 UnifiedNavigationEnv.step() 写入日志。

Usage examples:
    python tools/plot_tb_reward_log_walltime.py \
        --cbf logs/navigation_cbf/navigation_cbf_20250825_222719/\
events.out.tfevents.1756186041.yangl-amber.476802.0 \
        --naive logs/navigation_naive/navigation_naive_20250825_231444/\
events.out.tfevents.1756188887.yangl-amber.507008.0 \
        --tag "Mean episode reward_log" \
        --out logs/plots/mean_episode_reward_log_relative_time.png

If --tag is omitted, the script will try to locate a matching tag by heuristics
that search for case-insensitive matches containing "Mean" and "reward_log".

Dependencies:
  - matplotlib (required)
  - tensorboard (preferred) or tensorflow (fallback)
Install (if needed):
  pip install matplotlib tensorboard
  # or fallback
  pip install tensorflow
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker  # ADD THIS
from matplotlib.lines import Line2D  # NEW: for proxy legend handles

# make the font times new roman
plt.rcParams["font.family"] = "Times New Roman"

# ----------------------------- Readers ---------------------------------

def _try_import_event_accumulator():
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
        return EventAccumulator
    except Exception:
        return None

def _try_import_tf_summary_iterator():
    try:
        import tensorflow as tf  # type: ignore
        return tf.compat.v1.train.summary_iterator
    except Exception:
        return None

def read_scalars_with_event_accumulator(
    event_path: str,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """
    Returns a dict: tag -> list of (wall_time, step, value)
    """
    EventAccumulator = _try_import_event_accumulator()
    if EventAccumulator is None:
        raise ImportError(
            "tensorboard is not installed (EventAccumulator missing)"
        )

    # Size guidance keeps all scalars
    ea = EventAccumulator(
        event_path,
        size_guidance={
            'scalars': 0,  # 0 -> load all
        },
    )
    ea.Reload()

    tags = ea.Tags().get("scalars", [])
    result: Dict[str, List[Tuple[float, int, float]]] = {}
    for tag in tags:
        scalars = ea.Scalars(tag)
        result[tag] = [(s.wall_time, s.step, float(s.value)) for s in scalars]
    return result

def read_scalars_with_tf_iterator(
    event_path: str,
) -> Dict[str, List[Tuple[float, int, float]]]:
    summary_iterator = _try_import_tf_summary_iterator()
    if summary_iterator is None:
        raise ImportError(
            "tensorflow is not installed (summary_iterator missing)"
        )

    data: Dict[str, List[Tuple[float, int, float]]] = {}
    for event in summary_iterator(event_path):
        if not event.summary.value:
            continue
        for v in event.summary.value:
            # Only scalar values
            # v.simple_value holds float for scalar summaries
            if hasattr(v, "simple_value"):
                tag = v.tag
                wall_time = float(event.wall_time)
                step = int(event.step)
                val = float(v.simple_value)
                data.setdefault(tag, []).append((wall_time, step, val))
    return data

def read_all_scalars(
    event_path: str,
) -> Dict[str, List[Tuple[float, int, float]]]:
    """Attempt to read scalars using tensorboard EventAccumulator,
    else tensorflow iterator."""
    try:
        return read_scalars_with_event_accumulator(event_path)
    except Exception:
        # Fallback to TF summary_iterator
        return read_scalars_with_tf_iterator(event_path)

# ----------------------------- Tag Selection ---------------------------

def choose_tag(
    available: Sequence[str], desired: Optional[str]
) -> Tuple[str, List[str]]:
    """
    Choose the best matching tag.
    Returns (chosen_tag, tried_candidates) where tried_candidates shows
    precedence tried.
    """
    tags_set = set(available)
    tried: List[str] = []

    def _norm(s: str) -> str:
        return s.strip().lower()

    if desired:
        # Exact
        if desired in tags_set:
            return desired, [desired]
        # Case-insensitive exact
        for t in available:
            if _norm(t) == _norm(desired):
                tried.append(desired)
                return t, tried

    # Heuristics for typical names
    candidates = [
        "Train/mean_reward",
    ]

    # 1) Try exact then case-insensitive exact
    for c in candidates:
        tried.append(c)
        if c in tags_set:
            return c, tried
        for t in available:
            if _norm(t) == _norm(c):
                return t, tried

    # 2) Substring search prioritizing both "mean" and "reward_log"
    substr_priorities = [
        ("mean", "reward_log"),
        ("reward_log",),
        ("mean",),
    ]
    lower_tags = [t.lower() for t in available]
    for subs in substr_priorities:
        for i, lt in enumerate(lower_tags):
            if all(sub in lt for sub in subs):
                return available[i], tried

    # Fallback: if only one scalar tag exists, use it
    if len(available) == 1:
        return available[0], tried

    raise KeyError(
        "Could not identify a scalar tag for Mean episode reward_log.\n"
        + f"Available scalar tags ({len(available)}):\n  - "
        + "\n  - ".join(sorted(available))
    )

def find_tag_generic(
    available: Sequence[str],
    desired: Optional[str],
    substrings: Optional[List[Tuple[str, ...]]] = None,
) -> str:
    """Find a tag by exact/case-insensitive match; then by substrings.
    Raises KeyError if not found.
    """
    tags_set = set(available)

    def _norm(s: str) -> str:
        return s.strip().lower()

    if desired:
        if desired in tags_set:
            return desired
        for t in available:
            if _norm(t) == _norm(desired):
                return t

    if substrings:
        lower_tags = [t.lower() for t in available]
        for subs in substrings:
            for i, lt in enumerate(lower_tags):
                if all(sub in lt for sub in subs):
                    return available[i]

    raise KeyError(
        "Could not find desired tag. Available: "
        + ", ".join(sorted(available))
    )

# ----------------------------- Plotting --------------------------------

@dataclass
class Series:
    label: str
    times_sec: List[float]  # x-axis values (steps)
    values: List[float]

def to_series(
    scalars: Dict[str, List[Tuple[float, int, float]]],
    label: str,
    desired_tag: Optional[str],
) -> Tuple[Series, str, List[str]]:
    tags = list(scalars.keys())
    chosen, tried = choose_tag(tags, desired_tag)
    entries = scalars[chosen]
    # Sort by step to ensure monotonic x
    entries.sort(key=lambda x: x[1])
    # Use training steps on x-axis
    times_sec = [float(step) for _, step, _ in entries]
    values = [val for _, _, val in entries]
    return (
        Series(label=label, times_sec=times_sec, values=values),
        chosen,
        tried,
    )

def plot_series(
    series_list: List[Series],
    title: str,
    out_main_path: Optional[str] = None,
    out_summary_path: Optional[str] = None,
    yscale: str = "symlog",
    linthresh: float = 1e-2,
    ylim: Optional[Tuple[float, float]] = None,
    collision_ylim: Optional[Tuple[float, float]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    smoothing: float = 0.6,
    shade_raw: bool = True,
    shade_window: int = 21,
    shade_percentiles: Tuple[float, float] = (5.0, 95.0),
    show_raw_line: bool = False,
    show_summary: bool = True,
    summary_json: Optional[str] = None,
    summary_kind: str = "stacked",
    max_steps: Optional[float] = 1500.0,
    collisions: Optional[Dict[str, Series]] = None,
    collisions_smooth: Optional[float] = None,
    ieee_onecol: bool = False,
    x_tick_step: Optional[float] = None,
    x_max_ticks: Optional[int] = None,
) -> None:
    # helper to ensure .pdf extension
    def _ensure_pdf(path: str) -> str:
        base, _ = os.path.splitext(path)
        return base + ".pdf"

    # NEW: choose a "nice" tick step (1/2/5 * 10^k)
    def _nice_step(raw_step: float) -> float:
        if raw_step <= 0:
            return 1.0
        exp = np.floor(np.log10(raw_step))
        frac = raw_step / (10 ** exp)
        if frac <= 1:
            nice = 1.0
        elif frac <= 2:
            nice = 2.0
        elif frac <= 5:
            nice = 5.0
        else:
            nice = 10.0
        return float(nice * (10 ** exp))

    # Font sizes and figure sizes
    if ieee_onecol:
        fs_label, fs_tick, fs_legend, fs_title = 10, 10, 10, 10
        main_size = (3.5, 2.3)
        sum_size = (3.5, 2.6)  # CHANGED: give more height to summary axes
    else:
        fs_label, fs_tick, fs_legend, fs_title = 12, 12, 12, 12
        main_size = (16, 7)
        sum_size = (18, 7.2)   # CHANGED: slightly taller summary axes

    # set a smaller tick-label padding (points)
    tick_pad = 0.5 if ieee_onecol else 2.0
    # ---------------- Main time-series figure ----------------
    fig_main, ax_main = plt.subplots(1, 1, figsize=main_size)
    fig_main.set_constrained_layout(True)
    plt.sca(ax_main)
    colors_by_label: Dict[str, str] = {}
    # collect dashed collision line handles by method label
    coll_handle_map: Dict[str, object] = {}
    
    for s in series_list:
        if not s.times_sec:
            continue
        x = np.asarray(s.times_sec)
        y = np.asarray(s.values, dtype=float)

        # Limit to max_steps if provided
        if max_steps is not None:
            mask = x <= float(max_steps)
            if not np.any(mask):
                # nothing to plot for this series under the step cap
                continue
            x = x[mask]
            y = y[mask]

        # Optional raw line (thin, faint)
        line_color = None
        if show_raw_line:
            raw_line, = plt.plot(
                x,
                y,
                linestyle='-',
                color=None,
                alpha=0.35,
                linewidth=0.8,
            )
            line_color = raw_line.get_color()

        # Compute smoothed values using EMA similar to TensorBoard
        if 0.0 < smoothing < 1.0 and len(y) > 1:
            smoothed = np.copy(y)
            for i in range(1, len(y)):
                smoothed[i] = (
                    smoothing * smoothed[i - 1]
                    + (1.0 - smoothing) * y[i]
                )
        else:
            smoothed = y

        # Plot smoothed line (main, with label)
        main_line, = plt.plot(
            x,
            smoothed,
            # marker='o',
            linestyle='-',
            markersize=1.0,
            linewidth=1.5,
            label=s.label,
        )
        if line_color is None:
            line_color = main_line.get_color()
        # Normalize color to hex string for consistent typing
        colors_by_label[s.label] = mcolors.to_hex(main_line.get_color())

        # Shade envelope derived from raw values (percentile window)
        if shade_raw and len(y) > 2:
            k = max(3, int(shade_window) if shade_window is not None else 21)
            # ensure odd window for symmetric center
            if k % 2 == 0:
                k += 1
            half = k // 2
            p_lo, p_hi = shade_percentiles
            lo_env = np.empty_like(y)
            hi_env = np.empty_like(y)
            n = len(y)
            for i in range(n):
                a = max(0, i - half)
                b = min(n, i + half + 1)
                window_vals = y[a:b]
                lo_env[i] = np.percentile(window_vals, p_lo)
                hi_env[i] = np.percentile(window_vals, p_hi)
            plt.fill_between(
                x,
                lo_env,
                hi_env,
                color=line_color,
                alpha=0.15,
                linewidth=0,
            )

    plt.xlabel("Step", fontsize=fs_label)
    plt.ylabel("Reward", fontsize=fs_label)
    plt.title(title, fontsize=fs_title)
    ax_main.set_ylabel("Reward", fontsize=fs_label, labelpad=10)

    # Apply requested Y scale.
    # symlog is default to handle negative/zero reward values gracefully.
    if yscale == "symlog":
        plt.yscale("symlog", linthresh=linthresh)
    else:
        plt.yscale(yscale)

    # Apply Y limits for primary axis (rewards) if provided
    if ylim is not None:
        plt.ylim(ylim)

    # Apply X limit and set sparse ticks
    # Prefer max_steps; else use xlim; else skip custom ticks
    desired_max_ticks = (
        x_max_ticks if x_max_ticks is not None
        else (6 if ieee_onecol else 12)
    )
    if max_steps is not None:
        plt.xlim(right=float(max_steps))
        max_x = float(max_steps)
        step = (
            float(x_tick_step)
            if x_tick_step is not None
            else _nice_step(max_x / max(1, desired_max_ticks - 1))
        )
        ticks = np.arange(0.0, max_x + 0.5 * step, step)
        plt.xticks(ticks, fontsize=fs_tick)
    elif xlim is not None:
        plt.xlim(left=float(xlim[0]), right=float(xlim[1]))
        max_x = float(xlim[1])
        step = (
            float(x_tick_step)
            if x_tick_step is not None
            else _nice_step(max_x / max(1, desired_max_ticks - 1))
        )
        ticks = np.arange(float(xlim[0]), max_x + 0.5 * step, step)
        plt.xticks(ticks, fontsize=fs_tick)
    else:
        # Fallback: only set tick font size
        plt.xticks(fontsize=fs_tick)

    # Plot collisions on secondary y-axis with independent limits
    if collisions is not None and len(collisions) > 0:
        ax_coll = ax_main.twinx()
        ax_coll.set_ylabel("Collisions", fontsize=fs_label, labelpad=10)  # add labelpad
        COLL_SCALE = 4096.0  # scale collisions by 4096
         
        for s in series_list:
            if s.label not in collisions:
                continue
            sc = collisions[s.label]
            if not sc.times_sec:
                continue
            xc = np.asarray(sc.times_sec)
            yc = np.asarray(sc.values, dtype=float)
            
            # Limit to max steps
            if max_steps is not None:
                m2 = xc <= float(max_steps)
                if np.any(m2):
                    xc = xc[m2]
                    yc = yc[m2]
                else:
                    continue
            if not xc.size:
                continue
                
            # Smooth collisions with dedicated factor (or fallback)
            sm_c = (
                collisions_smooth
                if collisions_smooth is not None
                else smoothing
            )
            if 0.0 < sm_c < 1.0 and len(yc) > 1:
                ycs = np.copy(yc)
                for i in range(1, len(yc)):
                    ycs[i] = sm_c * ycs[i - 1] + (1.0 - sm_c) * yc[i]
            else:
                ycs = yc

            # Scale values by 4096 before plotting
            ycs = ycs * COLL_SCALE

            # Plot collisions without scaling on secondary axis
            coll_line, = ax_coll.plot(
                xc,
                ycs,
                linestyle='--',
                linewidth=1.5,
                color=colors_by_label.get(s.label, None),
                alpha=0.8,
                label=f"{s.label} (collisions)",
            )
            # store handle for collisions legend
            coll_handle_map[s.label] = coll_line

        # Apply Y limits for secondary axis (collisions) if provided
        if collision_ylim is not None:
            ax_coll.set_ylim(collision_ylim[0] * COLL_SCALE, collision_ylim[1] * COLL_SCALE)
        else:
            ax_coll.set_ylim(0.0, 1.0 * COLL_SCALE)

        # Force a draw to ensure all transforms are updated
        fig_main.canvas.draw()
        
        # Get all left axis tick positions and map to right axis
        left_ticks = ax_main.get_yticks()
        
        # Convert left ticks from data to display coordinates
        left_tick_points = np.column_stack([np.zeros_like(left_ticks), left_ticks])
        display_coords = ax_main.transData.transform(left_tick_points)
        
        # Map display coordinates to right axis data coordinates
        right_data_coords = ax_coll.transData.inverted().transform(display_coords)
        right_ticks = right_data_coords[:, 1]
        
        # Filter out any ticks that would be outside the right axis limits
        right_ymin, right_ymax = ax_coll.get_ylim()
        valid_ticks = right_ticks[(right_ticks >= right_ymin) & (right_ticks <= right_ymax)]
        
        # Use all valid ticks, or fall back to evenly spaced if needed
        if len(valid_ticks) >= 2:
            ax_coll.set_yticks(valid_ticks)
        else:
            # Fallback if mapping fails
            ax_coll.set_yticks(np.linspace(right_ymin, right_ymax, 4))

        # Format right-axis tick labels as integers (rounded)
        ax_coll.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))

        ax_coll.tick_params(axis='y', labelsize=fs_tick)
        # ax_coll.legend(fontsize=fs_legend, loc="upper right")  # moved to bottom combined legend

    # Set axis labels
    ax_main.set_xlabel("Step")
    ax_main.set_ylabel("Reward")
    ax_main.set_title(title)

    # If we created a right axis for collisions, collect it; else just the main axis
    axes = [ax_main]
    if collisions is not None and len(collisions) > 0:
        axes.append(ax_coll)

    # Apply consistent font sizes to all text objects, and do it at the end
    for ax in axes:
        # ticks
        ax.tick_params(axis="both", which="major", labelsize=fs_tick, pad=tick_pad)
        ax.tick_params(axis="both", which="minor", labelsize=fs_tick, pad=tick_pad)
        # axis labels and title
        ax.xaxis.label.set_size(fs_label)
        ax.yaxis.label.set_size(fs_label)
        ax.title.set_fontsize(fs_title)
        # ensure consistent font weight/family (optional)
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontweight("normal")

    # If you prefer rcParams, set them once near the start of plot_series:
    # plt.rcParams.update({
    #     "font.size": fs_tick,
    #     "axes.labelsize": fs_label,
    #     "axes.titlesize": fs_title,
    #     "xtick.labelsize": fs_tick,
    #     "ytick.labelsize": fs_tick,
    # })

    plt.grid(True, alpha=0.3)
    # Build a combined legend (methods only, in desired order) below x labels
    preferred_order = ["Hybrid", "Reward Only", "Filter Only", "Naive"]
    handles_all, labels_all = ax_main.get_legend_handles_labels()
    method_handles = {}
    for h, l in zip(handles_all, labels_all):
        if "(collisions)" in l:
            continue
        if l in preferred_order and l not in method_handles:
            method_handles[l] = h
    handles = [method_handles[l] for l in preferred_order if l in method_handles]
    labels = [l for l in preferred_order if l in method_handles]

    # Build thin proxy dashed handles for collisions so dashes are visible in the legend
    coll_order = ["Hybrid", "Reward Only", "Filter Only", "Naive"]
    coll_labels = [f"{l} Collisions" for l in coll_order if l in coll_handle_map]
    dash_pat = (5, 3) if ieee_onecol else (6, 3)
    proxy_lw = 1.1 if ieee_onecol else 1.1
    coll_handles = [
        Line2D(
            [0], [0],
            linestyle='--',
            dashes=dash_pat,
            linewidth=proxy_lw,
            color=colors_by_label.get(l, "black"),
            alpha=0.9,
        )
        for l in coll_order if l in coll_handle_map
    ]

    # Arrange into 3 rows within one legend box (4-2-2):
    #   Row 1: 4 solid method items
    #   Row 2: first 2 dashed collision items + 2 spacers
    #   Row 3: next 2 dashed collision items + 2 spacers
    def _spacer_handle():
        return Line2D([0], [0], linestyle='none', linewidth=0.0, alpha=0.0)

    ncol = 4

    # Row 1: methods (pad to 4)
    row1_h, row1_l = list(handles), list(labels)
    while len(row1_h) < ncol:
        row1_h.append(_spacer_handle())
        row1_l.append("")

    # Row 2: first two collisions (pad to 4)
    row2_h, row2_l = coll_handles[:2], coll_labels[:2]
    while len(row2_h) < ncol:
        row2_h.append(_spacer_handle())
        row2_l.append("")

    # Row 3: next two collisions (pad to 4)
    row3_h, row3_l = coll_handles[2:4], coll_labels[2:4]
    while len(row3_h) < ncol:
        row3_h.append(_spacer_handle())
        row3_l.append("")

    # Column-major interleave so top row shows solids left-to-right
    combined_handles, combined_labels = [], []
    for c in range(ncol):
        combined_handles.extend([row1_h[c], row2_h[c], row3_h[c]])
        combined_labels.extend([row1_l[c], row2_l[c], row3_l[c]])

    fig_main.legend(
        combined_handles,
        combined_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.23),  # room for 3 rows
        ncol=ncol,
        fontsize=fs_legend,
        frameon=True,
        borderaxespad=0.0,
        handlelength=1.8,
        handletextpad=0.3,
        columnspacing=0.6,
        borderpad=0.2,
        labelspacing=0.2,
    )

    plt.yticks(fontsize=fs_tick)
    # For interactive display, add a tiny bottom margin so the legend is visible on screen.
    if out_main_path is None:
        # Enough room for a three-row single legend box
        fig_main.subplots_adjust(bottom=(0.24 if ieee_onecol else 0.22))
    # Save main as PDF (enforce .pdf and format)
    if out_main_path:
        out_main_path = _ensure_pdf(out_main_path)
        os.makedirs(os.path.dirname(out_main_path), exist_ok=True)
        fig_main.savefig(
            out_main_path,
            dpi=150,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        # PNG companion: convenient for previewing and inserting into a report.
        png_main_path = os.path.splitext(out_main_path)[0] + ".png"
        fig_main.savefig(
            png_main_path,
            dpi=220,
            format="png",
            bbox_inches="tight",
            pad_inches=0.02,
        )
        print(f"Saved main plot to: {out_main_path}")
        print(f"Saved PNG preview to: {png_main_path}")
    else:
        plt.show()

    # ---------------- Summary bar figure ----------------
    if show_summary:
        try:
            summary = load_summary_data(summary_json)
            fig_sum, ax_sum = plt.subplots(1, 1, figsize=sum_size)
            fig_sum.set_constrained_layout(True)  # NEW: maximize drawable area

            # draw and get category legend handles for bottom placement
            cat_handles, cat_labels = draw_summary_axes(
                ax_sum,
                summary,
                kind=summary_kind,
                fs_label=fs_label,
                fs_tick=fs_tick,
                fs_title=fs_title,
                fs_legend=fs_legend,
            )

            # Place legend outside; bbox_inches='tight' on save will include it
            fig_sum.legend(
                cat_handles,
                cat_labels,
                loc="lower center",
                bbox_to_anchor=(0.62, -0.1),  # moved below to avoid covering x ticks
                ncol=len(cat_labels),
                fontsize=fs_legend,
                frameon=True,
                borderaxespad=0.0,
                # tighter legend spacing
                handlelength=0.8,
                handletextpad=0.3,
                columnspacing=0.6,
                borderpad=0.2,
                labelspacing=0.2,
            )

            # For interactive display, add a tiny bottom margin so the legend is visible
            if out_summary_path is None:
                fig_sum.subplots_adjust(bottom=(0.12 if ieee_onecol else 0.10))

            # Save summary as PDF (derive if not provided)
            if out_summary_path:
                out_summary_path = _ensure_pdf(out_summary_path)
            elif out_main_path:
                base, _ = os.path.splitext(out_main_path)
                out_summary_path = base + "_summary.pdf"
            if out_summary_path:
                os.makedirs(os.path.dirname(out_summary_path), exist_ok=True)
                fig_sum.savefig(
                    out_summary_path,
                    dpi=150,
                    format="pdf",
                    bbox_inches="tight",
                    pad_inches=0.06,  # increased to avoid cutting off rightmost tick label
                )
                print(f"Saved summary plot to: {out_summary_path}")
            else:
                plt.show()
        except Exception as e:
            print(f"Warning: failed to draw summary panel: {e}")

# ----------------------------- CLI -------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Mean episode reward_log vs training steps from"
            " TensorBoard event files"
        )
    )
    parser.add_argument(
        "--cbf",
        type=str,
        default=(
            "logs/navigation_cbf/navigation_cbf_20260313_003437/"
            "events.out.tfevents.1773387279.yangl-amber.2671586.0"
        ),
        help="Path to CBF run event file",
    )
    parser.add_argument(
        "--naive",
        type=str,
        default=(
            "logs/navigation_naive/navigation_naive_20260313_012712/"
            "events.out.tfevents.1773390434.yangl-amber.2713352.0"
        ),
        help="Path to naive/baseline run event file",
    )
    parser.add_argument(
        "--soft-cbf",
        dest="soft_cbf",
        type=str,
        default=(
            "logs/navigation_reward_only/navigation_reward_only_20260313_015617/"
            "events.out.tfevents.1773392178.yangl-amber.2732388.0"
        ),
        help="Path to Soft CBF run event file",
    )
    # NEW: Only CBF run
    parser.add_argument(
        "--only-cbf",
        dest="only_cbf",
        type=str,
        default=(
            "logs/navigation_filter_only/navigation_filter_only_20260313_010047/"
            "events.out.tfevents.1773388848.yangl-amber.2696230.0"
        ),
        help="Path to Only CBF run event file",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="Mean episode reward_log",
        help=(
            "Exact tag to read (heuristics will try close matches if not "
            "found)"
        ),
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Deprecated: use --out-main. If provided, used as main output (pdf enforced).",
    )
    parser.add_argument(
        "--out-main",
        type=str,
        default="logs/plots/mean_episode_reward_log_steps.pdf",
        help="PDF path for main time-series plot",
    )
    parser.add_argument(
        "--out-summary",
        type=str,
        default="logs/plots/mean_episode_reward_log_summary.pdf",
        help="PDF path for summary bar plot",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Mean episode reward vs. steps",
        help="Plot title",
    )
    parser.add_argument(
        "--yscale",
        type=str,
        choices=["linear", "log", "symlog"],
        default="linear",
        help="Y-axis scale (default: symlog; use log only if values > 0)",
    )
    parser.add_argument(
        "--linthresh",
        type=float,
        default=1e-2,
        help="Linear range around zero for symlog (ignored for linear/log)",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=[200.0, 2200.0],
        help="Primary y-axis limits (rewards) as two floats: YMIN YMAX (default: 200 2200)",
    )
    parser.add_argument(
        "--collision-ylim",
        type=float,
        nargs=2,
        metavar=("YMIN", "YMAX"),
        default=[0.0, 0.01],
        help="Secondary y-axis limits (collisions) as two floats: YMIN YMAX (default: 0.0 1.0)",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        nargs=2,
        metavar=("XMIN", "XMAX"),
        default=[0.0, 1550.0],
        help="X-axis limits as two floats: XMIN XMAX (default: 0.0 1550.0)",
    )
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.8,
        help=(
            "Exponential smoothing factor in [0,1). 0=no smoothing, "
            "closer to 1=more smoothing"
        ),
    )
    parser.add_argument(
        "--shade-window",
        type=int,
        default=21,
        help=(
            "Window size (number of points) for raw envelope percentiles "
            "(default: 21)"
        ),
    )
    parser.add_argument(
        "--shade-percentiles",
        type=float,
        nargs=2,
        metavar=("PLOW", "PHIGH"),
        default=[5.0, 95.0],
        help=(
            "Percentiles for raw shading envelope (default: 5 95). "
            "Use 0 100 for min-max."
        ),
    )
    parser.add_argument(
        "--no-shade",
        action="store_true",
        help="Disable shading of raw (unsmoothed) envelope",
    )
    parser.add_argument(
        "--show-raw-line",
        action="store_true",
        help="Also draw a faint raw line behind the smoothed curve",
    )
    parser.add_argument(
        "--summary",
        dest="show_summary",
        action="store_true",
        help="Show lower-right inset with success/failure breakdown",
    )
    parser.add_argument(
        "--no-summary",
        dest="show_summary",
        action="store_false",
        help="Disable summary inset",
    )
    parser.set_defaults(show_summary=True)
    parser.add_argument(
        "--summary-json",
        type=str,
        default=None,
        help=(
            "Optional JSON file with summary metrics. If omitted, uses "
            "built-in defaults from the prompt"
        )
    )
    parser.add_argument(
        "--summary-kind",
        type=str,
        choices=["stacked", "grouped"],
        default="stacked",
        help=(
            "Inset chart type: 'stacked' (100% stacked bars, default) or "
            "'grouped' (grouped bars by category)"
        ),
    )
    parser.add_argument(
        "--max-steps",
        type=float,
        default=1500.0,
        help="Only plot data up to and including this step (default: 1500)",
    )
    parser.add_argument(
        "--collisions-tag",
        type=str,
        default="Episode/collided_obstacle",
        help="Tag name for collisions metric to overlay",
    )
    parser.add_argument(
        "--no-collisions",
        action="store_true",
        help="Disable plotting the collisions overlay",
    )
    parser.add_argument(
        "--smooth-collisions",
        type=float,
        default=0.9,
        help=(
            "Exponential smoothing factor for collisions in [0,1). "
            "Overrides --smooth for collisions only (default: 0.9)"
        ),
    )
    parser.add_argument(
        "--ieee-onecol",
        dest="ieee_onecol",
        action="store_true",
        help="Format figures for IEEE one-column width (3.5in) with small fonts",
    )
    # NEW: control x-axis tick density
    parser.add_argument(
        "--x-tick-step",
        type=float,
        default=None,
        help="Explicit step between x ticks (overrides auto).",
    )
    parser.add_argument(
        "--x-max-ticks",
        type=int,
        default=None,
        help="Max number of x ticks for auto spacing (default: 6 if --ieee-onecol else 12).",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    series: List[Series] = []
    collisions_map: Dict[str, Series] = {}

    runs = [
        (args.cbf, "Hybrid"),
        (args.soft_cbf, "Reward Only"),
        (args.only_cbf, "Filter Only"),  # NEW
        (args.naive, "Naive"),
    ]

    for path, label in runs:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Warning: event file not found: {path}")
            continue
        try:
            scalars = read_all_scalars(path)
        except Exception as e:
            print(f"Failed to read scalars from {path}: {e}")
            continue

        try:
            s, chosen_tag, tried = to_series(
                scalars, label=label, desired_tag=args.tag
            )
        except Exception as e:
            available = ", ".join(sorted(scalars.keys()))
            print(
                f"Could not find tag in {path}: {e}\n"
                f"Available tags: {available}"
            )
            continue

        print(f"{label}: using tag '{chosen_tag}'. Tried candidates: {tried}")
        print(f"{label}: {len(s.times_sec)} points loaded")
        series.append(s)

        # Try to load collisions series for the same run
        if not args.no_collisions:
            try:
                coll_tag = find_tag_generic(
                    list(scalars.keys()),
                    args.collisions_tag,
                    substrings=[("collid",), ("collision",)],
                )
                coll_entries = scalars[coll_tag]
                coll_entries.sort(key=lambda x: x[1])
                coll_steps = [float(step) for _, step, _ in coll_entries]
                coll_vals = [val for _, _, val in coll_entries]
                collisions_map[label] = Series(
                    label=label, times_sec=coll_steps, values=coll_vals
                )
                print(
                    f"{label}: overlay tag '{coll_tag}' loaded "
                    f"({len(coll_steps)} points)"
                )
            except Exception as e:
                print(f"{label}: collisions overlay not added: {e}")

    if not series:
        print("No data loaded. Exiting.")
        return

    title = args.title

    # Derive outputs: prefer explicit new args; fallback to --out
    out_main = args.out_main or args.out
    out_summary = args.out_summary  # if None, plot_series will derive from main

    plot_series(
        series,
        title=title,
        out_main_path=out_main,
        out_summary_path=out_summary,
        yscale=args.yscale,
        linthresh=args.linthresh,
        ylim=(args.ylim[0], args.ylim[1]) if args.ylim else None,
        collision_ylim=(args.collision_ylim[0], args.collision_ylim[1]) if args.collision_ylim else None,
        xlim=(args.xlim[0], args.xlim[1]) if args.xlim else None,
        smoothing=max(0.0, min(0.9999, args.smooth)),
        shade_raw=not args.no_shade,
        shade_window=args.shade_window,
        shade_percentiles=(args.shade_percentiles[0], args.shade_percentiles[1]),
        show_raw_line=args.show_raw_line,
        show_summary=args.show_summary,
        summary_json=args.summary_json,
        summary_kind=args.summary_kind,
        max_steps=args.max_steps,
        collisions=collisions_map if not args.no_collisions else None,
        collisions_smooth=args.smooth_collisions,
        ieee_onecol=args.ieee_onecol,
        # NEW: pass tick controls
        x_tick_step=args.x_tick_step,
        x_max_ticks=args.x_max_ticks,
    )

# ----------------------------- Summary Inset --------------------------

def default_summary_data() -> List[Dict[str, float | str]]:
    """Default metrics based on the provided results.

    Percentages are in [0, 100]. Keys per entry:
      label, success_pct, fail_obstacle_pct, fail_wall_pct, fail_timeout_pct
    """
    return [
        {
            "label": "CBF (no filter)",
            "success_pct": 92.70,
            "fail_obstacle_pct": 6.00,
            "fail_wall_pct": 0.7,
            "fail_timeout_pct": 0.6,
        },
        {
            "label": "CBF (filter)",
            "success_pct": 99.00,
            "fail_obstacle_pct": 0.20,
            "fail_wall_pct": 0.00,
            "fail_timeout_pct": 0.80,
        },
        {
            "label": "Soft CBF",
            "success_pct": 91.90,
            "fail_obstacle_pct": 2.20,
            "fail_wall_pct": 0.50,
            "fail_timeout_pct": 5.40,
        },
        {
            "label": "Naive",
            "success_pct": 51.40,
            "fail_obstacle_pct": 40.50,
            "fail_wall_pct": 0.60,
            "fail_timeout_pct": 7.50,
        },
        {
            "label": "Only CBF",
            "success_pct": 57.90,
            "fail_obstacle_pct": 29.50,
            "fail_wall_pct": 1.70,
            "fail_timeout_pct": 10.90,
        },
    ]

def load_summary_data(path: Optional[str]) -> List[Dict[str, float | str]]:
    if not path:
        return default_summary_data()
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "methods" not in obj:
        raise ValueError("summary JSON must be an object with key 'methods'")
    methods = obj["methods"]
    if not isinstance(methods, list):
        raise ValueError("'methods' must be a list")
    out: List[Dict[str, float | str]] = []
    for m in methods:
        out.append(
            {
                "label": m.get("label", "Unnamed"),
                "success_pct": float(m.get("success_pct", 0.0)),
                "fail_obstacle_pct": float(m.get("fail_obstacle_pct", 0.0)),
                "fail_wall_pct": float(m.get("fail_wall_pct", 0.0)),
                "fail_timeout_pct": float(m.get("fail_timeout_pct", 0.0)),
            }
        )
    return out

def draw_summary_axes(
    ax,
    summary: List[Dict[str, float | str]],
    kind: str = "stacked",
    fs_label: int = 12,
    fs_tick: int = 12,
    fs_title: int = 12,
    fs_legend: int = 12,
) -> tuple[list, list]:
    # Reorder and filter methods to desired rows; include both CBF variants
    desired_rows = ["CBF", "CBF (no filter)", "Soft CBF", "Only CBF", "Naive"]  # CHANGED

    def _norm(s: str) -> str:
        return s.strip().lower()

    # Find indexes for each desired row
    idx_cbf_no_filter = next(
        (
            i
            for i, m in enumerate(summary)
            if _norm(str(m["label"])) in ("cbf (no filter)", "cbf (without filter)", "cbf no filter")
        ),
        None,
    )
    idx_cbf_filter = next(
        (i for i, m in enumerate(summary) if _norm(str(m["label"])) == "cbf (filter)"),
        None,
    )
    # Fallback for generic "CBF" not marked as "(no filter)"
    if idx_cbf_filter is None:
        idx_cbf_filter = next(
            (
                i
                for i, m in enumerate(summary)
                if _norm(str(m["label"])).startswith("cbf")
                and _norm(str(m["label"])) not in ("cbf (no filter)", "cbf (without filter)", "cbf no filter")
            ),
            None,
        )

    pick_idx = {
        "CBF (no filter)": idx_cbf_no_filter,
        "CBF": idx_cbf_filter,
        "Soft CBF": next((i for i, m in enumerate(summary) if _norm(str(m["label"])) == "soft cbf"), None),
        "Only CBF": next((i for i, m in enumerate(summary) if _norm(str(m["label"])) == "only cbf"), None),
        "Naive": next((i for i, m in enumerate(summary) if _norm(str(m["label"])) == "naive"), None),
    }

    rows = []
    for name in desired_rows:
        idx = pick_idx.get(name)
        if idx is None:
            continue
        rows.append(
            (
                name,
                float(summary[idx]["success_pct"]),
                float(summary[idx]["fail_obstacle_pct"]),
                float(summary[idx]["fail_wall_pct"]),
                float(summary[idx]["fail_timeout_pct"]),
            )
        )
    if not rows:
        # fallback to original behavior order
        labels = [str(m["label"]) for m in summary]
        success = np.array([float(m["success_pct"]) for m in summary])
        obst = np.array([float(m["fail_obstacle_pct"]) for m in summary])
        wall = np.array([float(m["fail_wall_pct"]) for m in summary])
        tout = np.array([float(m["fail_timeout_pct"]) for m in summary])
        rows = list(zip(labels, success, obst, wall, tout))

    # Normalize each row to sum 100 for visualization
    data = np.array([[r[1], r[2], r[3], r[4]] for r in rows], dtype=float)
    totals = data.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(totals > 0, 100.0 / totals, 0.0)
    data = data * scale

    y_labels = [r[0] for r in rows]
    n = len(y_labels)
    # Plot first item at the top without inverting axis by using descending y positions
    y = np.arange(n - 1, -1, -1)  # CHANGED

    width = 0.7
    colors = {
        "success": "#2ca02c",
        "obstacle": "#d62728",
        "wall": "#ff7f0e",
        "timeout": "#1f77b4",
    }

    left0 = np.zeros(n)
    b_succ = ax.barh(y, data[:, 0], height=width, left=left0, label="Success", color=colors["success"])
    left1 = left0 + data[:, 0]
    b_obst = ax.barh(y, data[:, 1], height=width, left=left1, label="Obstacle", color=colors["obstacle"])
    left2 = left1 + data[:, 1]
    b_wall = ax.barh(y, data[:, 2], height=width, left=left2, label="Wall", color=colors["wall"])
    left3 = left2 + data[:, 2]
    b_tout = ax.barh(y, data[:, 3], height=width, left=left3, label="Timeout", color=colors["timeout"])

    # Annotate success % inside the success segment, if wide enough
    for i, rect in enumerate(b_succ):
        val = data[i, 0]
        if val >= 8.0:
            ax.text(
                rect.get_x() + rect.get_width() - 1.0,
                rect.get_y() + rect.get_height() / 2.0,
                f"{val:.1f}%",
                ha="right",
                va="center",
                fontsize=fs_tick,
                color="white",
                fontweight="bold",
            )

    ax.set_xlim(0, 100.5)  # add small right margin so "100" is fully visible
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels, fontsize=fs_tick)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=fs_tick)
    ax.set_xlabel("Percentage (%)", fontsize=fs_label)
    ax.set_title("Result breakdown (%)", fontsize=fs_title)
    ax.grid(True, axis="x", alpha=0.2)
    # tighten tick label spacing on summary axes
    _pad = 0.5 if fs_tick <= 10 else 2.0
    ax.tick_params(axis="both", which="major", pad=_pad)
    ax.tick_params(axis="both", which="minor", pad=_pad)
    # Return handles/labels for bottom figure legend
    return [b_succ[0], b_obst[0], b_wall[0], b_tout[0]], ["Success", "Obstacle", "Wall", "Timeout"]
if __name__ == "__main__":
    main()
