#!/bin/bash
# 评测 Dual checkpoint，并保留运行时 CBF-filter。
python test.py --config dual_runtime_filter --headless $@
