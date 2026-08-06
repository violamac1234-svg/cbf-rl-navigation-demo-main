"""论文 Table III 对应的训练与部署配置。

Training profiles define how a checkpoint is produced. Evaluation profiles
separate the training method from runtime filtering and dynamics disturbance,
which is required to express all 12 configurations in Table III.
"""

from dataclasses import asdict, dataclass
from typing import Dict


@dataclass(frozen=True)
class MethodConfig:
    """定义一个 checkpoint 在训练阶段使用的两个 CBF 组件。"""
    use_cbf_reward: bool
    use_training_filter: bool


@dataclass(frozen=True)
class EvaluationConfig:
    """定义评测时加载哪个 checkpoint，以及部署环境是否过滤/加扰动。"""
    training_method: str
    use_runtime_filter: bool
    use_domain_randomization: bool


# 四种消融训练方法：安全奖励和训练时过滤器的 2x2 组合。
TRAINING_METHODS: Dict[str, MethodConfig] = {
    "nominal": MethodConfig(False, False),
    "reward_only": MethodConfig(True, False),
    "filter_only": MethodConfig(False, True),
    "dual": MethodConfig(True, True),
}


# 运行时过滤器必须与训练时过滤器分开，否则无法评测“训练有过滤、部署无过滤”。
TABLE_III_CONFIGS: Dict[str, EvaluationConfig] = {
    "nominal": EvaluationConfig("nominal", False, False),
    "dual_runtime_filter": EvaluationConfig("dual", True, False),
    "dual_no_runtime_filter": EvaluationConfig("dual", False, False),
    "reward_only": EvaluationConfig("reward_only", False, False),
    "filter_only_runtime_filter": EvaluationConfig("filter_only", True, False),
    "filter_only_no_runtime_filter": EvaluationConfig("filter_only", False, False),
    "nominal_dr": EvaluationConfig("nominal", False, True),
    "dual_dr_runtime_filter": EvaluationConfig("dual", True, True),
    "dual_dr_no_runtime_filter": EvaluationConfig("dual", False, True),
    "reward_only_dr": EvaluationConfig("reward_only", False, True),
    "filter_only_dr_runtime_filter": EvaluationConfig("filter_only", True, True),
    "filter_only_dr_no_runtime_filter": EvaluationConfig("filter_only", False, True),
}


def method_config(name: str) -> dict:
    if name not in TRAINING_METHODS:
        raise ValueError(f"Unknown training method: {name}")
    return asdict(TRAINING_METHODS[name])


def evaluation_config(name: str) -> dict:
    if name not in TABLE_III_CONFIGS:
        raise ValueError(f"Unknown evaluation configuration: {name}")
    return asdict(TABLE_III_CONFIGS[name])
