"""FIVES 数据集评测器：完整 FIVES 论文推荐指标集。"""

from evaluators.base import BaseEvaluator


class FIVESEvaluator(BaseEvaluator):
    """FIVES 评测器，在默认指标基础上增加 clDice。"""

    def get_metric_names(self):
        return ["dice", "iou", "mcc", "bm", "se", "sp", "pr", "auc", "hd95", "asd", "cldice"]
