#!/bin/bash
# 评测 Nominal checkpoint，不使用运行时过滤器。
python test.py --config nominal --headless $@
