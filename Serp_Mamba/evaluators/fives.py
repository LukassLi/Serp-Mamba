"""FIVES 数据集评测器：完整 FIVES 论文推荐指标集。"""

from evaluators.base import BaseEvaluator


class FIVESEvaluator(BaseEvaluator):
    """FIVES 评测器，包含 Dice、IoU、MCC、BM、SE、SP、PR、AUC、HD95 九项指标。"""

    def get_metric_names(self):
        return ["dice", "iou", "mcc", "bm", "se", "sp", "pr", "auc", "hd95"]
