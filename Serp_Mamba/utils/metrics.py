#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
metrics.py - 公共指标工具，统一所有评价指标的计算逻辑。

供 test.py、evaluate.py、val_2D.py 等模块复用，消除重复代码。
"""

import numpy as np
from medpy import metric as medpy_metric
from skimage.morphology import skeletonize


# ── 通用单值指标 ──

def calc_dice(pred, gt):
    """Dice 系数 (F1)"""
    if pred.sum() == 0:
        return 0.0
    return medpy_metric.binary.dc(pred, gt)


def calc_iou(pred, gt):
    """IoU / Jaccard 指数"""
    if pred.sum() == 0:
        return 0.0
    return medpy_metric.binary.jc(pred, gt)


def calc_hd95(pred, gt):
    """Hausdorff Distance 95th percentile"""
    if pred.sum() > 0 and gt.sum() > 0:
        return medpy_metric.binary.hd95(pred, gt)
    return 0.0


def calc_asd(pred, gt):
    """Average Surface Distance"""
    if pred.sum() > 0 and gt.sum() > 0:
        return medpy_metric.binary.asd(pred, gt)
    return 0.0


def calc_auc(pred_prob, gt):
    """AUC (需要 softmax 概率，不是二值预测)"""
    from sklearn.metrics import roc_auc_score
    gt_flat = gt.flatten().astype(int)
    prob_flat = pred_prob.flatten()
    if len(np.unique(gt_flat)) < 2:
        return 0.0
    return roc_auc_score(gt_flat, prob_flat)


# ── 基于混淆矩阵的指标族 ──

def confusion_matrix(pred, gt):
    """返回 (TP, FP, TN, FN)"""
    TP = ((pred == 1) & (gt == 1)).sum()
    FP = ((pred == 1) & (gt == 0)).sum()
    TN = ((pred == 0) & (gt == 0)).sum()
    FN = ((pred == 0) & (gt == 1)).sum()
    return TP, FP, TN, FN


def calc_sensitivity(pred, gt):
    """Sensitivity / Recall / TPR — 血管检出率"""
    TP, FP, TN, FN = confusion_matrix(pred, gt)
    return TP / (TP + FN) if (TP + FN) > 0 else 0.0


def calc_specificity(pred, gt):
    """Specificity / TNR — 背景准确率"""
    TP, FP, TN, FN = confusion_matrix(pred, gt)
    return TN / (TN + FP) if (TN + FP) > 0 else 0.0


def calc_precision(pred, gt):
    """Precision — 预测血管精确率"""
    TP, FP, TN, FN = confusion_matrix(pred, gt)
    return TP / (TP + FP) if (TP + FP) > 0 else 0.0


def calc_mcc(pred, gt):
    """Matthews Correlation Coefficient"""
    TP, FP, TN, FN = confusion_matrix(pred, gt)
    denom = np.sqrt(float(TP + FP) * float(TP + FN) * float(TN + FP) * float(TN + FN))
    return ((TP * TN) - (FP * FN)) / denom if denom > 0 else 0.0


def calc_bm(pred, gt):
    """Bookmaker Informedness (Youden's J) = TPR + TNR - 1"""
    return calc_sensitivity(pred, gt) + calc_specificity(pred, gt) - 1


# ── 兼容旧接口（val_2D.py 的 calculate_metric_percase 用到） ──

def calculate_metric_percase(pred, gt):
    """返回 (Dice, IoU, HD95, ASD)"""
    dc = calc_dice(pred, gt)
    jc = calc_iou(pred, gt)
    hd = calc_hd95(pred, gt)
    asd = calc_asd(pred, gt)
    return dc, jc, hd, asd


# ── PyTorch 张量级 Dice（用于训练 loss） ──

def dice(input, target, ignore_index=None):
    smooth = 1.
    iflat = input.clone().view(-1)
    tflat = target.clone().view(-1)
    if ignore_index is not None:
        mask = tflat == ignore_index
        tflat[mask] = 0
        iflat[mask] = 0
    intersection = (iflat * tflat).sum()
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)


# ── 拓扑指标 ──

def calc_cldice(pred, gt):
    """clDice（中心线 Dice）：衡量预测与标签在骨架层面的拓扑重叠。

    计算方式：先提取骨架，再分别计算骨架被对方掩码覆盖的比例，
    最后取几何平均。值越高说明血管连通性保持越好。
    """
    if pred.sum() == 0 or gt.sum() == 0:
        return 0.0
    pred_sk = skeletonize(pred.astype(bool))
    gt_sk = skeletonize(gt.astype(bool))
    # 预测骨架有多少落在 GT 掩码内
    tprec = pred_sk[gt.astype(bool)].sum() / pred_sk.sum() if pred_sk.sum() > 0 else 0.0
    # GT 骨架有多少落在预测掩码内
    tsens = gt_sk[pred.astype(bool)].sum() / gt_sk.sum() if gt_sk.sum() > 0 else 0.0
    if tprec + tsens == 0:
        return 0.0
    return 2.0 * tprec * tsens / (tprec + tsens)


# ── 批量计算接口 ──

# 注册表：指标名 → 计算函数
METRIC_REGISTRY = {
    "dice": calc_dice,
    "iou": calc_iou,
    "mcc": calc_mcc,
    "bm": calc_bm,
    "se": calc_sensitivity,
    "sp": calc_specificity,
    "pr": calc_precision,
    "auc": calc_auc,       # 特殊：需要 pred_prob 而非 pred
    "hd95": calc_hd95,
    "asd": calc_asd,
    "cldice": calc_cldice,
}

# 需要 pred_prob 而非 pred 二值图的指标
PROB_BASED_METRICS = {"auc"}


def compute_metrics(pred, gt, metric_names, pred_prob=None):
    """
    统一接口：根据指标名列表批量计算。

    Args:
        pred: 二值预测 (numpy array, 0/1)
        gt: 二值标签 (numpy array, 0/1)
        metric_names: 指标名列表，如 ["dice", "iou", "se", "auc"]
        pred_prob: softmax 概率图 (仅 AUC 等概率类指标需要)

    Returns:
        dict: {metric_name: float_value}
    """
    results = {}
    for name in metric_names:
        fn = METRIC_REGISTRY.get(name)
        if fn is None:
            raise ValueError(f"未知指标: {name}")
        if name in PROB_BASED_METRICS:
            if pred_prob is None:
                raise ValueError(f"指标 '{name}' 需要 pred_prob 参数")
            results[name] = fn(pred_prob, gt)
        else:
            results[name] = fn(pred, gt)
    return results
