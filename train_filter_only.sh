#!/bin/bash
# Filter Only：训练 rollout 执行 CBF-filter 动作，但不加入安全奖励。
python train.py --method filter_only --headless $@
