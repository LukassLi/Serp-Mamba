"""默认评测器：Dice, IoU, MCC, BM — 适用于 DRIVE / Fundus / PRIME-FP20 等通用数据集。"""

from evaluators.base import BaseEvaluator


class DefaultEvaluator(BaseEvaluator):
    """默认评测器，计算 Dice、IoU、MCC、BM 四项基础指标。"""

    def get_metric_names(self):
        return ["dice", "iou", "mcc", "bm"]
