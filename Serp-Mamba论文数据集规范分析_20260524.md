# Serp-Mamba 论文数据集规范 vs 项目实际配置分析

> 时间戳：2026-05-24

## Context

对照论文确认项目实际使用的数据集划分是否符合论文要求。

---

## 论文中的数据集与划分方式

论文使用 **3 个 UWF-SLO 数据集**，全部采用 **五折交叉验证（5-fold cross-validation）**：

| 数据集 | 图像数量 | 来源 | 评估方式 |
|--------|----------|------|----------|
| PRIME-FP20 | **15 张** | Optos California/200Tx 相机 | 五折交叉验证 |
| MU-VS Center A | **30 张** | 医疗中心 A | 五折交叉验证 |
| MU-VS Center B | **30 张** | 医疗中心 B | 五折交叉验证 |

### 关键论文原文

> Section III-C: "we conduct a five-fold cross-validation experiment to enhance the robustness and reliability of our findings."

> Section IV-A: "The PRIME-FP20 dataset includes 15 UWF-SLO images... Each image is accompanied by a binary mask delineating its vessels."

> Table I 中每个数据集的指标都带有 Std（标准差），这是五折交叉验证的结果格式。

### 论文训练参数

- 输入尺寸：1024 × 1024
- 优化器：Adam, lr=0.0001, weight_decay=0.0001
- 训练迭代：**12,000 次**
- 评估指标：Dice, IoU, MCC, BM
- GPU：NVIDIA Tesla V100 32GB

---

## 项目当前的 PRIME-FP20 配置

`configs/prime_fp20.yaml` 当前配置：

```
train root: PRIME-FP20-after-VAL1  →  training_image/ (15 张)
val root:   PRIME-FP20-after-VAL1  →  val_image/ (15 张)
test root:  PRIME-FP20-TEST/val1_test →  val_image/ (? 张)
```

`train.py` 默认参数：
- max_iterations: **16,000**（论文要求 12,000）
- patch_size: [1024, 1024] ✓
- base_lr: 0.0001 ✓

---

## 差异分析

### 问题 1：划分方式不符（最关键）

| 项目 | 论文要求 | 项目实际 |
|------|----------|----------|
| PRIME-FP20 总图数 | **15 张** | 训练 15 + 验证 15 + 测试 ? |
| 划分方式 | **五折交叉验证** | 固定 train/val/test 三分 |
| 验证方式 | 5 折取平均 ± 标准差 | 单次 train/val 训练 |

论文的 15 张图做五折交叉验证意味着：每折用 12 张训练、3 张验证，循环 5 次取平均。项目当前有 15 张训练图 + 15 张验证图，总共 30 张，与论文描述的 15 张不符。

### 问题 2：训练迭代数不符

- 论文：**12,000** iterations
- 项目默认：**16,000** iterations

### 问题 3：数据来源不明

- 论文明确 PRIME-FP20 只有 15 张 UWF-SLO 图像
- 项目中的 `training_image/`(15张) 和 `val_image/`(15张) 加起来 30 张，超出了论文描述
- `PRIME-FP20-TEST/val1_test/` 中的测试图数量和来源也不明确

---

## 结论

**项目当前的 PRIME-FP20 配置不符合论文要求。** 主要差异：

1. **划分方式**：论文用五折交叉验证（15 张图），项目用固定 train/val/test 划分（30+ 张）
2. **图像数量**：论文 PRIME-FP20 总共 15 张，项目训练集就有 15 张
3. **训练迭代数**：论文 12,000，项目默认 16,000

如果要严格复现论文结果，需要实现五折交叉验证的训练/评估流程。
