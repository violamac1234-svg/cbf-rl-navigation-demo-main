#!/bin/bash
# Dual/CBF-RL：同时启用安全奖励和训练时 CBF-filter。
python train.py --method dual --headless $@
