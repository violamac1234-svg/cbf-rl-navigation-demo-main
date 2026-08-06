"""训练低阶二维二连杆的四种 CBF-RL 消融配置。"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import random

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from arm2d.vec_env import Arm2DVecEnv
from experiment_configs import TRAINING_METHODS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=TRAINING_METHODS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--steps-per-env", type=int, default=32)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--run-label", default="")
    return parser.parse_args()


def make_training_config(args):
    config = {
        "seed": args.seed,
        "device": args.device,
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.8,
            "actor_hidden_dims": [64, 64],
            "critic_hidden_dims": [64, 64],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.003,
            "num_learning_epochs": 5,
            "num_mini_batches": 8,
            "learning_rate": 3.0e-4,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "num_steps_per_env": args.steps_per_env,
        "save_interval": args.save_interval,
        "empirical_normalization": False,
    }
    return config


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    method = TRAINING_METHODS[args.method]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{args.run_label}" if args.run_label else ""
    log_dir = Path("logs") / "arm2d" / args.method / f"{timestamp}_seed{args.seed}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        **vars(args),
        "use_cbf_reward": method.use_cbf_reward,
        "use_training_filter": method.use_training_filter,
        "model": "joint_velocity_kinematics",
    }
    (log_dir / "experiment_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    env = Arm2DVecEnv(args.num_envs, method, args.device, args.seed)
    train_cfg = make_training_config(args)
    print(f"method={args.method} envs={args.num_envs} iterations={args.iterations} device={args.device}")
    print(f"CBF reward={method.use_cbf_reward}, training filter={method.use_training_filter}")
    print(f"log_dir={log_dir.resolve()}")
    runner = OnPolicyRunner(env, deepcopy(train_cfg), str(log_dir), device=args.device)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)
    runner.save(str(log_dir / f"model_{args.iterations}.pt"))
    print("training complete")


if __name__ == "__main__":
    main()
