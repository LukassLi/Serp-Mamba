"""
evaluators — 数据集评测器注册表。

根据 YAML 配置中的 evaluator 字段自动选择对应的评测器。
新增数据集只需：① 在 evaluators/ 下新建 .py ② 在此注册。
"""

from evaluators.default import DefaultEvaluator
from evaluators.fives import FIVESEvaluator

# 评测器注册表：名称 → 类
EVALUATOR_REGISTRY = {
    "default": DefaultEvaluator,
    "fives": FIVESEvaluator,
}


def get_evaluator(config):
    """
    根据配置获取评测器实例。

    优先读 config["evaluator"]，否则回退到 "default"。
    """
    evaluator_name = config.get("evaluator", "default")
    cls = EVALUATOR_REGISTRY.get(evaluator_name)
    if cls is None:
        raise ValueError(
            f"未知评测器: {evaluator_name}，可选: {list(EVALUATOR_REGISTRY.keys())}"
        )
    return cls()
