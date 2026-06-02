"""
BaseEvaluator — 评测器基类，定义统一接口。

子类只需实现 get_metric_names() 指定要计算的指标列表，
默认的 evaluate_single / aggregate / format_* 方法即可工作。
"""

import numpy as np
from utils.metrics import compute_metrics


class BaseEvaluator:
    """评测器基类。"""

    def get_metric_names(self):
        """返回本评测器计算的指标名列表（子类必须实现）。"""
        raise NotImplementedError

    def evaluate_single(self, prediction, label, pred_prob=None):
        """
        评估单张图片。

        默认实现调用 compute_metrics，子类可覆盖以自定义计算逻辑。
        """
        return compute_metrics(
            prediction, label, self.get_metric_names(), pred_prob=pred_prob
        )

    def aggregate(self, per_image_results):
        """
        汇总逐图结果 → 均值 ± 标准差。

        Args:
            per_image_results: list[dict]，每个元素是 evaluate_single 的返回值

        Returns:
            dict: {metric_name: (mean, std)}
        """
        perf = {}
        for name in self.get_metric_names():
            values = [r[name] for r in per_image_results]
            perf[name] = (np.mean(values), np.std(values))
        return perf

    def format_per_image(self, case_name, metrics):
        """格式化单图结果为字符串。"""
        parts = [f"{k}={v:.6f}" for k, v in metrics.items()]
        return f"  {case_name}: " + ", ".join(parts) + "\n"

    def format_summary(self, perf, model_name, split, num_samples):
        """格式化汇总结果为字符串。"""
        lines = [
            f"model_name = {model_name}\n",
            f"split = {split}\n",
            f"num_samples = {num_samples}\n",
        ]
        for name, (mean_val, std_val) in perf.items():
            lines.append(f"{name} = mean-sd = {mean_val}-{std_val}\n")
        return "".join(lines)
