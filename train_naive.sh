#!/bin/bash
# Nominal：无安全奖励、无训练时过滤器。
python train.py --method nominal --headless $@
