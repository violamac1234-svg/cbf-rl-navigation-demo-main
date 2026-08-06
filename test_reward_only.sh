#!/bin/bash
# 评测 Reward Only checkpoint，不使用运行时过滤器。
python test.py --config reward_only --headless $@
