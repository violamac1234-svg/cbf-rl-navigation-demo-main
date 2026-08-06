"""Four CBF-RL ablations for the G1 planar-obstacle experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodConfig:
    use_cbf_reward: bool
    use_training_filter: bool
    use_runtime_filter: bool = False


METHODS = {
    "nominal": MethodConfig(False, False),
    "reward_only": MethodConfig(True, False),
    "filter_only": MethodConfig(False, True),
    "dual": MethodConfig(True, True),
}

