#!/bin/bash
# Reward Only：只启用论文式 (22)-(23) 的 CBF 安全奖励。
python train.py --method reward_only --headless $@
