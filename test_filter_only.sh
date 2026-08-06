#!/bin/bash
# 评测 Filter Only checkpoint，并保留运行时 CBF-filter。
python test.py --config filter_only_runtime_filter --headless $@
