"""默认评测器：覆盖视网膜血管分割领域通用指标集。"""

from evaluators.base import BaseEvaluator


class DefaultEvaluator(BaseEvaluator):
    """默认评测器，包含 Dice、IoU、MCC、BM、SE、SP、PR、AUC、HD95、ASD、clDice。"""

    def get_metric_names(self):
        return ["dice", "iou", "mcc", "bm", "se", "sp", "pr", "auc", "hd95", "asd", "cldice"]
