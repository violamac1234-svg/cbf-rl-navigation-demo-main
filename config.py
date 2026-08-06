"""二维单积分器环境、PPO 和实验运行器的集中配置。"""
import torch
import os
import numpy as np # Added for observation size calculation

# --- Environment Specific Parameters ---
# These might be overridden by command-line args if using Hydra later
ENV_ID = 'NaiveNavigation-v0' # Use the registered environment ID
NUM_OBSTACLES = 3
WORLD_SIZE = 10.0
# Calculate expected flattened observation size
# robot_pos(2) + obstacles(NUM_OBSTACLES*3) + goal_pos(2) + last_velocity(2)
# The policy observes only the closest obstacle: [x, y, radius].
FLATTENED_OBS_SIZE = 2 + 3 + 2 + 2

# --- Configuration Definition ---
cfg = {
    'seed': 42, # Changed seed for consistency
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',

    'env': {
        'env_id': ENV_ID,
        'num_envs': 4096, # Number of parallel environments for training (adjust based on resources)
        # Environment specific args (can be passed to gym.make via runner config)
        'world_size': WORLD_SIZE,
        'num_obstacles': NUM_OBSTACLES,
        'max_episode_steps': 1000, # Default max steps per episode
        'dt': 0.05, # Time step for the environment
        # 训练脚本在 DR 模式下把 noise_level 覆盖为论文使用的 0.2。
        'noise_level': 0.0,
        # CBF 条件：∇h(q)^T v + alpha*h(q) >= 0。
        'cbf_alpha': 1.0,
        # 论文 Table II 中两个安全奖励项的共同权重。
        'cbf_reward_weight': 100.0,
        # 动作修正奖励指数项的尺度 sigma（论文为 0.5）。
        'cbf_sigma': 0.5,
        # Add other NaiveNavigationEnv args if needed (e.g., robot_radius)
        # 'make_env_func' is removed - we'll use env_id and wrap externally
    },

    'policy': {
        # 这些二维网络尺寸来自演示代码，并非论文明确给出的参数。
        'class_name': 'ActorCritic', # Assuming rsl-rl uses this naming convention
        'actor_hidden_dims': [32, 32], # Example MLP size
        'critic_hidden_dims': [32, 32], # Example MLP size
        'activation': 'relu',
    },

    'algorithm': {
        # PPO 超参数；论文未完整列出，复现时应在结果元数据中保留。
        'class_name': 'PPO',
        'gamma': 0.99,
        'lam': 0.95,
        'learning_rate': 3.e-4, # May need tuning
        # 'schedule': 'fixed',
        'entropy_coef': 0.005, # May need tuning
        'clip_param': 0.2,
        'num_learning_epochs': 10,
        'num_mini_batches': 8, # Adjust based on num_envs * num_steps_per_env
    },

    'runner': {
        # --- Training Specific ---
        'run_name': 'navigation', # Updated run name
        'experiment_name': 'navigation', # Updated experiment name
        'max_iterations': 1500, # Increased iterations for harder task
        'num_steps_per_env': 48, # Horizon per env per iteration (adjust)
        'save_interval': 100, # Save more frequently
        'log_interval': 20,

        # --- Checkpoint Loading ---
        'resume': False,
        'load_run': -1, # -1 loads the latest run directory matching run_name
        'checkpoint': -1, # -1 loads the latest checkpoint in that directory

        # --- Testing Specific ---
        'test_episodes': 1000, # More episodes for evaluation
        'render_test': False # Set True to watch testing
    }
}

# --- Runner config flattening for OnPolicyRunner compatibility ---
# Copy runner settings to top-level and ensure empirical_normalization is set
cfg['runner'].setdefault('empirical_normalization', False)
cfg['runner'].setdefault('log_reward_components', True)
cfg.update(cfg['runner'])

# --- Log Directory Logic ---
def get_log_dir(base_log_path='logs'):
    """Generates the base path for logging and saving models for the current config."""
    # This path points to the directory containing potentially multiple runs (timestamps)
    experiment_log_dir = os.path.join(base_log_path, cfg['runner']['experiment_name'], cfg['runner']['run_name'])
    return experiment_log_dir
