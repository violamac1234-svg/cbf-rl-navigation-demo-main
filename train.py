"""训练四种二维导航消融方法，并记录可复现实验元数据。"""

import os
from datetime import datetime
import argparse # Import argparse
import json
import random
import numpy as np
import torch

# Import configuration
from config import cfg, get_log_dir, FLATTENED_OBS_SIZE

from nav_env.unified_navigation_env import UnifiedNavigationEnv
from experiment_configs import TRAINING_METHODS

# Attempt to import rsl_rl components
try:
    from rsl_rl.runners import OnPolicyRunner
    _RSL_RL_AVAILABLE = True
except ImportError:
    print("Warning: rsl-rl components not found. Ensure rsl-rl is installed correctly.")
    print("Training script requires rsl-rl to function.")
    _RSL_RL_AVAILABLE = False

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Train navigation agent with rsl-rl")
    parser.add_argument('--method', '--env', dest='method', choices=TRAINING_METHODS, default='nominal')
    parser.add_argument('--domain-randomization', action='store_true', help='Use paper DR scale (20% of maximum velocity)')
    parser.add_argument('--seed', type=int, default=cfg['seed'])
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode (no GUI).")
    return parser.parse_args()

def train():
    """Initializes and runs the rsl-rl training process."""
    if not _RSL_RL_AVAILABLE:
        print("Cannot proceed without rsl-rl installed.")
        return

    args = parse_args() # Parse arguments
    # 同时固定所有实际使用的随机数发生器；CUDA 多卡也使用同一实验种子。
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg['seed'] = args.seed
    # method_cfg 决定安全奖励和训练时安全过滤器是否启用。
    method_cfg = TRAINING_METHODS[args.method]

    # Add environment type and timestamp to run name and experiment name for better tracking
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')    
    run_suffix = f"{args.method}{'_dr' if args.domain_randomization else ''}"
    cfg['runner']['run_name'] = f"{cfg['runner']['run_name']}_{run_suffix}_{timestamp}"
    cfg['runner']['experiment_name'] = f"{cfg['runner']['experiment_name']}_{run_suffix}"

    print("--- Starting Training ---")
    print(f"Run Name: {cfg['runner']['run_name']}")
    print(f"Experiment Name: {cfg['runner']['experiment_name']}")
    print(f"Device: {cfg['device']}")
    print(f"Method: {args.method}")
    print(f"Observation Size (Flattened): {FLATTENED_OBS_SIZE}")
    print(f"Number of Parallel Environments: {cfg['env']['num_envs']}")
    print(f"Max Training Iterations: {cfg['runner']['max_iterations']}")

    # --- Log Directory ---
    base_log_dir = get_log_dir()
    os.makedirs(base_log_dir, exist_ok=True)
    print(f"Base log directory for TensorBoard logs: {base_log_dir}")

    # --- Environment Setup ---
    num_envs = cfg['env']['num_envs']
    print(f"Run name updated to: {cfg['runner']['run_name']}")
    env_kwargs = {k: v for k, v in cfg['env'].items() if k not in ['env_id', 'num_envs', 'noise_level']}

    # Instantiate the selected environment
    # 论文 DR：每个时间步加入 0.2*v_max 的独立标准正态速度扰动。
    vec_env = UnifiedNavigationEnv(
        num_envs=num_envs,
        noise_level=0.2 if args.domain_randomization else 0.0,
        use_cbf_action_filtering=method_cfg.use_training_filter,
        use_cbf_reward_penalty=method_cfg.use_cbf_reward,
        **env_kwargs
    )
    print(f"--> Using UnifiedNavigationEnv as vectorized environment.")
    print(f"--> use_training_filter: {method_cfg.use_training_filter}")
    print(f"--> use_cbf_reward: {method_cfg.use_cbf_reward}")
    print(f"--> use_domain_randomization: {args.domain_randomization}")
    # checkpoint 名称本身不足以证明实验设置，因此同时保存机器可读配置。
    with open(os.path.join(base_log_dir, "experiment_config.json"), "w", encoding="utf-8") as stream:
        json.dump({"method": args.method, "seed": args.seed,
                   "use_cbf_reward": method_cfg.use_cbf_reward,
                   "use_training_filter": method_cfg.use_training_filter,
                   "use_domain_randomization": args.domain_randomization,
                   "noise_level": 0.2 if args.domain_randomization else 0.0},
                  stream, indent=2)
    
    # add the ability to change run_name to be timestamped
    if not args.headless:
        vec_env.render_mode = "human"
    

    try:
        runner = OnPolicyRunner(
            env=vec_env,
            train_cfg=cfg,
            log_dir=base_log_dir,
            device=cfg['device']
        )
        print("\nOnPolicyRunner initialized successfully.")
        # print("--> Using NaiveNavigationEnv as vectorized environment.") # Removed redundant print
        print("--> Check TensorBoard for rollout and reward statistics.\n")

    except Exception as e:
         print(f"\nError initializing OnPolicyRunner: {e}")
         print("Troubleshooting Tips:")
         print(" - Ensure the configuration structure (config.py) matches rsl-rl expectations.")
         print(" - Verify network input/output dimensions match environment spaces.")
         print(" - Check rsl-rl documentation for Runner initialization and required config fields.")
         print(" - Make sure 'nav_env' is correctly installed and importable.\n")
         return

    print(f"Starting training for {cfg['runner']['max_iterations']} iterations...")
    print(f"*** Monitor TensorBoard logs in the subdirectory created within: {base_log_dir} ***")
    print(f"*** To view: run 'tensorboard --logdir {base_log_dir}' (or point to specific run) ***")
    print("*** Look for tags like 'Loss/policy_loss', 'Loss/value_loss', 'rollout/ep_rew_mean', 'rollout/reward_goal_mean', etc. ***\n")
    try:
        runner.learn(num_learning_iterations=cfg['runner']['max_iterations'],
                     init_at_random_ep_len=True)
        print("\n--- Training finished ---")
    except KeyboardInterrupt:
         print("\n--- Training interrupted by user ---")
    except Exception as e:
        print(f"\n--- An error occurred during training: {e} ---")
    finally:
        print("Performing cleanup (if any)...")
        print("Cleanup complete.")

if __name__ == '__main__':
    train()
