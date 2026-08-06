# CBF Learning Demo

This repository contains a demonstration of combining Reinforcement Learning (RL) with Control Barrier Functions (CBF) for robot navigation, using `rsl_rl` and IsaacGym-like vectorized environments. 

## 2-DOF Arm Migration

The repository also contains a Windows-tested low-order CBF-RL implementation for a planar two-link arm.
See the Chinese [run guide](./arm2d/RUN_GUIDE.md) for environment setup, smoke tests, four-method
training, checkpoint layout, and Fig.3/Fig.4 plotting commands. The mathematical model and source-code
mapping are documented in [LOW_ORDER_MODEL.md](./arm2d/LOW_ORDER_MODEL.md).

## Installation

The project uses Conda to manage its dependencies. First, ensure you have Conda installed on your system.
The `environment.yml` file is the original Linux environment snapshot for the navigation project. It contains
Linux-specific build identifiers and is not the Windows environment used by the Arm2D migration. For Arm2D on
native Windows, follow [arm2d/RUN_GUIDE.md](./arm2d/RUN_GUIDE.md) instead of the command below.

Create and activate the conda environment by running:
```bash
conda env create -f environment.yml
conda activate cbf_learning
```

## Environments Setup

The project provides several environment configurations through the `UnifiedNavigationEnv` by passing different arguments:
- **Naive**: Standard RL without CBF.
- **CBF (Hybrid)**: RL with CBF action filtering and CBF reward penalties.
- **Filter Only**: RL with CBF action filtering but no reward penalties.
- **Reward Only (Soft CBF)**: RL with CBF reward penalties but no action filtering.

## Training

You can train the agent using the provided bash scripts. For example, to run headless training:
```bash
# Train Naive 
./train_naive.sh

# Train CBF
./train_cbf.sh

# Train Filter Only
./train_filter_only.sh

# Train Reward Only
./train_reward_only.sh
```

Alternatively, you can run `train.py` directly with custom command line flags. For example:
```bash
python train.py --method dual --headless
python train.py --method dual --domain-randomization --headless
```

On Windows, `run_all_training.ps1` performs the first reproduction stage and
trains only Nominal, Reward Only, Filter Only, and Dual. Domain randomization
is disabled in this stage. Each run writes an `experiment_config.json` beside
its checkpoints.

After the four-method ablation is complete, use `run_dr_training.ps1` for the
separate 20% dynamics-disturbance experiment. The 12 Table III deployment
configurations are a later evaluation stage rather than part of this default
training batch.

Training logs and checkpoints are automatically saved to `logs/navigation_<env_type>/`.

## Testing / Evaluation

After training, evaluate the learned policy using the test scripts:
```bash
./test_naive.sh
./test_cbf.sh
./test_filter_only.sh
./test_reward_only.sh
```

Or run `test.py` directly. For example:
```bash
python test.py --config dual_no_runtime_filter --episodes 1000 --headless
```

`run_table_iii.ps1` evaluates all 12 configurations from Table III. Results
are written as per-episode CSV files and JSON summaries under
`results/table_iii/` by default.

The test script automatically finds the latest run directory for the specified environment type and loads the latest checkpoint. It will play out the scenario visually (unless run with `--headless`) and output success rate and failure reasons (e.g., collisions).

## Plotting TensorBoard Logs

The repository includes a script to plot the `Mean episode reward` over training steps from TensorBoard event files. You'll need the paths to your run's event files. 

Example usage:
```bash
python plot_tb_reward_log_steps.py \
    --cbf logs/navigation_cbf/.../events.out.tfevents... \
    --naive logs/navigation_naive/.../events.out.tfevents... \
    --only-cbf logs/navigation_filter_only/.../events.out.tfevents... \
    --soft-cbf logs/navigation_reward_only/.../events.out.tfevents... \
    --out-main logs/plots/mean_episode_reward.pdf
```
This generates PDF plots of the training rewards and obstacle collisions over time. The main plot is saved to `logs/plots/mean_episode_reward.pdf` and a summary bar chart to `logs/plots/mean_episode_reward_summary.pdf`.
