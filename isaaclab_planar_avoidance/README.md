# G1 planar obstacle avoidance: first replication stage

This directory is the first, deliberately small IsaacLab replication stage for
the humanoid planar-obstacle experiment in *CBF-RL: Safety Filtering
Reinforcement Learning in Training with Control Barrier Functions*.

## What is implemented

- batched circular barrier `h = ||p_robot - p_obstacle|| - R_robot - R_obstacle`;
- the active-obstacle gradient;
- the single-constraint closed-form CBF projection;
- the paper's CBF reward (constraint violation plus intervention penalty);
- explicit Nominal, Reward Only, Filter Only, and Dual experiment switches;
- simulator-independent unit tests.

## Important replication boundary

The paper states that the G1 policy outputs 12 lower-body joint position
targets, while the planar CBF is written in the reduced two-dimensional base
velocity. It does not publish the implementation that lifts a filtered planar
velocity back to those 12 joint targets. Therefore this stage does **not**
pretend that filtering the 12-D action is solved.

The IsaacLab integration will use the official `Isaac-Velocity-Flat-G1-v0`
task as the locomotion baseline and will add a named control-interface adapter.
Two implementations can then be compared without contaminating the CBF math:

1. a command-space surrogate that filters the planar velocity command;
2. a learned or model-based reduced-to-full-order adapter, if the authors'
   implementation becomes available.

## Local verification

From the repository root, with the existing `cbf_learning` environment:

```powershell
python -m unittest discover -s isaaclab_planar_avoidance/tests -v
```

## IsaacLab baseline gate

Before adding obstacles, the following official task must train and play:

```powershell
isaaclab.bat -p scripts/reinforcement_learning/rsl_rl/train.py `
  --task Isaac-Velocity-Flat-G1-v0 --num_envs 128 --headless
```

Use 128 environments first on an 8 GB GPU. Increase only after measuring peak
VRAM. The paper's 4096-environment setting targeted an RTX 4090.

## Next integration steps

1. Pin the installed IsaacLab/Isaac Sim versions in experiment metadata.
2. Add one fixed cylinder per environment and expose its body-frame relative
   position to policy observations.
3. Add collision termination and success/CBF metrics.
4. Implement the named reduced-to-full-order control-interface adapter.
5. Run the four ablations with identical seeds and scenes.
