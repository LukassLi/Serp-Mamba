# Serp-Mamba 多数据集扩展实施计划

> 时间戳：2026-05-23

## Context

当前项目仅支持 PRIME-FP20 数据集，目录布局、文件名映射、图像模式等全部硬编码在 `BaseDataSets` 类和 `train.py`/`test.py` 中。为支持不同格式的视网膜血管分割数据集，需要引入轻量配置层，将数据集差异外化到 YAML 配置文件中。

核心原则：**最小修改范围，不触碰模型代码，不破坏已有功能**。

---

## 修改总览

| 文件 | 操作 | 说明 |
|------|------|------|
| `dataloaders/dataset_registry.py` | **新建** | ConfigDataSets 类 + 配置加载器（~90行） |
| `configs/prime_fp20.yaml` | **新建** | PRIME-FP20 配置（复现当前硬编码行为） |
| `configs/fundus.yaml` | **新建** | fundus_dataset 配置 |
| `configs/drive.yaml` | **新建** | DRIVE 数据集配置 |
| `train.py` | **修改** | ~15行：加 `--dataset` 参数，替换 BaseDataSets，参数化路径和 input_channels |
| `test.py` | **修改** | ~10行：同上模式 |
| `dataloaders/dataset.py` | 不改 | BaseDataSets 保留，向后兼容 |
| `val_2D.py` | 不改 | 已经参数化，不依赖数据集类 |
| `SerpMamba.py` | 不改 | 模型代码不动 |
| `utils/*` | 不改 | 通用工具，与数据集无关 |

---

## 1. 新建 `dataloaders/dataset_registry.py`

配置驱动的数据集类，替代 `BaseDataSets` 的调用点。

```python
# 核心内容：
def load_dataset_config(config_path: str) -> dict
    # 加载 YAML 配置，校验必填字段

def _transform_label_name(image_filename: str, transforms: list) -> str
    # 三种操作：replace, change_ext, append_suffix
    # 覆盖 PRIME-FP20 / DRIVE / CHASE_DB1 / STARE / HRF 的文件名映射

class ConfigDataSets(Dataset)
    # __init__(self, config: dict, split: str, transform=None)
    #   - split_map 映射 split 名到目录名
    #   - image_dir/label_dir 支持 {split} 占位符
    #   - image_ext 过滤文件
    #
    # _load_image(path) -> np.ndarray
    #   - 按 config["image_mode"] 加载（"L" 灰度 / "RGB" 彩色）
    #
    # _load_label(image_filename) -> np.ndarray
    #   - 调用 _transform_label_name 推导标签文件名
    #   - RGB 标签按 label_rgb_mode 转单通道
    #   - label_threshold 二值化
    #
    # __getitem__ 返回格式与 BaseDataSets 完全一致：
    #   {"image": tensor, "label": tensor, "name": str, "idx": int}
```

### 设计依据

- **为什么新建类而非修改 BaseDataSets**：`BaseDataSets` 的 `__getitem__` 方法（`dataset.py:50-90`）中，三个 split 分支各自硬编码了目录路径和文件名映射。若修改此类，会影响所有已有引用。新建 `ConfigDataSets` 是纯加法操作，零风险。
- **为什么返回格式一致**：训练循环 `train.py:218-221` 通过 `sampled_batch["image"]`、`sampled_batch["label"]`、`sampled_batch["name"]` 取数据；`val_2D.py:99` 的 `test_image_fast` 函数接收 `(image, label, net, classes, patch_size)` 参数。只要返回 dict 格式不变，下游无需改动。
- **为什么用 dict 传参而非文件路径**：调用方（`train.py`/`test.py`）加载配置，dataset 类接收 dict。这样 dataset 类不依赖文件系统，便于测试。

---

## 2. 新建 `configs/` 目录 + YAML 配置文件

### 配置 Schema

```yaml
name: "数据集名称"
description: "简述"

root_dir: "/path/to/dataset"          # 数据集根目录

# 目录布局：{split} 会被 split_map 映射后的值替换
image_dir: "{split}_image"            # 如 "training_image", "images"
label_dir: "{split}_label"

# split 名到实际目录名的映射
split_map:
  train: "training"                   # train -> training_image / training_label
  val: "val"
  test: "test"

# 文件匹配
image_ext: ".tif"                     # 按 extension 过滤图像文件
label_ext: ".png"

# 标签文件名推导（按顺序执行）
label_name_transform:
  - replace: ["Img", "Label"]
  - replace: ["tif", "png"]

# 图像属性
image_mode: "L"                       # PIL 加载模式：L(灰度) / RGB
input_channels: 1                     # 送入模型的通道数
label_threshold: 0                    # label[label > threshold] = 1
label_rgb_mode: "mean"                # RGB 标签转单通道：mean / green / first

# 训练默认超参（可被 argparse 覆盖）
patch_size: [1024, 1024]
batch_size: 1
max_iterations: 16000
base_lr: 0.0001
num_classes: 2
```

### `configs/prime_fp20.yaml`

复现当前硬编码行为，依据如下：
- `dataset.py:37-44`：目录命名规则为 `{split}_image/` 和 `{split}_label/`，split 取值 `"training"` / `"val"` / `"test"`
- `dataset.py:53,56`：图像用 `convert('L')` 加载灰度，标签用 `.replace('Img', 'Label').replace('tif', 'png')` 映射文件名
- `dataset.py:58-59`：RGB 标签用 `mean(axis=-1)` 转单通道，`label[label > 0] = 1` 二值化
- `train.py:174`：`SerpMamba(input_channels=1, ...)`

### `configs/fundus.yaml`

适配 `U_Mamba_main/data/fundus_dataset/` 的布局：
- 目录结构：`{split}/images/`、`{split}/vessel_masks/`（直接观察目录）
- 图像 `.jpg`，标签 `.png`，同名不同扩展名
- `image_mode: "RGB"`（.jpg 彩色图像），`input_channels: 1`（转灰度送入模型）

---

## 3. 修改 `train.py`

### 3a. 新增参数（`train.py:50`，argparse 部分）
```python
parser.add_argument("--dataset", type=str, default="prime_fp20",
                    help="Dataset config name or path to YAML")
parser.add_argument("--output_dir", type=str, default=None,
                    help="Output dir for checkpoints/logs")
```
依据：当前通过 `--root_path` 传入数据集路径（`train.py:25-26`），但路径只是数据集属性之一。新增 `--dataset` 将所有数据集属性统一管理。

### 3b. 加载配置（`train.py:51`，parse_args 之后）
```python
from dataloaders.dataset_registry import load_dataset_config, ConfigDataSets

cfg_path = args.dataset
if not cfg_path.endswith('.yaml'):
    cfg_path = os.path.join(os.path.dirname(__file__), 'configs', cfg_path + '.yaml')
dataset_cfg = load_dataset_config(cfg_path)
```

### 3c. 替换数据集实例化（`train.py:187-189`）
```python
# Before:
db_train = BaseDataSets(base_dir=args.root_path, split="train", transform=RandomGenerator(args.patch_size))
db_val = BaseDataSets(base_dir=args.root_path, split="val")

# After:
patch_size = dataset_cfg.get("patch_size", args.patch_size)
db_train = ConfigDataSets(config=dataset_cfg, split="train", transform=RandomGenerator(patch_size))
db_val = ConfigDataSets(config=dataset_cfg, split="val")
```
依据：`train.py:187` 是 `BaseDataSets` 的唯一调用点，替换后接口不变（返回格式一致），训练循环 `train.py:218-231` 无需改动。

### 3d. 参数化 input_channels（`train.py:174`）
```python
# Before: model = SerpMamba(input_channels=1, ...)
# After:  model = SerpMamba(input_channels=dataset_cfg.get("input_channels", 1), ...)
```
依据：`SerpMamba.__init__` 的 `input_channels` 参数（`SerpMamba.py:1067`）决定第一层卷积的输入通道。当前硬编码为 1（灰度），RGB 数据集需要不同值。

### 3e. 参数化输出路径（`train.py:207,327`）
```python
# Before (line 327):
snapshot_path = "/home/lishh237/Serp-Mamba/Serp_Mamba/{}_{}".format(args.exp, args.model)
# Before (line 207):
writer = SummaryWriter("/home/lishh237/Serp-Mamba/Serp_Mamba/tf-logs/")

# After:
snapshot_path = args.output_dir or os.path.join(os.path.dirname(__file__),
    "experiments", dataset_cfg["name"])
writer = SummaryWriter(os.path.join(snapshot_path, "tf-logs"))
```
依据：`train.py:327` 和 `train.py:207` 硬编码了远程服务器路径 `/home/lishh237/...`，本地环境无法使用。改为相对路径 + 数据集名称自动生成。

### 3f. 超参可被 config 覆盖（已实现）
优先使用 argparse 显式传入值，否则用 config 中的默认值：
```python
base_lr = (args.base_lr if args.base_lr != parser.get_default("base_lr")
           else dataset_cfg.get("base_lr", args.base_lr))
# 同理 max_iterations, batch_size, num_classes
```
依据：不同数据集可能需要不同的训练参数（如 DRIVE 分辨率小可增大 batch_size、HRF 分辨率大需更多 iteration）。

### 不改动
- `unet_config`（`train.py:53-127`）和 `other_kwargs`（`train.py:131-137`）：模型架构参数，非数据集相关
- 训练循环逻辑（`train.py:216-310`）：loss 计算、优化器、验证逻辑与数据集无关
- `RandomGenerator`（`dataset.py:161-191`）：transform 逻辑通用，不依赖数据集格式

---

## 4. 修改 `test.py`

### 4a. 新增参数（`test.py:57`）
```python
parser.add_argument("--dataset", type=str, default="prime_fp20")
parser.add_argument("--checkpoint_dir", type=str, default=None,
                    help="Directory containing .pth checkpoints")
```

### 4b. 同 train.py 加载配置

### 4c. 替换 BaseDataSets（`test.py:230`）
```python
# Before: db_val = BaseDataSets(base_dir=FLAGS.root_path, split="val")
# After:  db_val = ConfigDataSets(config=dataset_cfg, split="val")
```

### 4d. 参数化 input_channels（`test.py:244`）
```python
# Before: SerpMamba(input_channels=1, ...)
# After:  SerpMamba(input_channels=dataset_cfg.get("input_channels", 1), ...)
```

### 4e. 参数化路径（`test.py:214,233`）
```python
# Before (line 233): folder_path = "/home/lishh237/Serp-Mamba/Serp_Mamba/val1/"
# After:
folder_path = FLAGS.checkpoint_dir or os.path.join(os.path.dirname(__file__),
    "experiments", dataset_cfg["name"])

# Before (line 214): image.save('/home/lishh237/.../output2_image.png')
# After: image.save(os.path.join(folder_path, 'output_image.png'))
```
依据：`test.py:233` 硬编码了权重路径，`test.py:214` 硬编码了输出保存路径，均为远程服务器路径，本地不可用。

---

## 5. 添加新数据集的步骤

1. 准备数据集目录（任意布局均可）
2. 在 `configs/` 下创建 YAML 配置文件，描述目录结构、文件名映射、图像属性
3. 训练：`python train.py --dataset my_dataset`
4. 测试：`python test.py --dataset my_dataset --checkpoint_dir experiments/MyDataset`

---

## 6. 回归测试方案

目标：验证修改后对 PRIME-FP20 数据集的训练和测试流程与修改前完全一致。

### 阶段一：修改前基线采集

在**未修改**的代码中，临时添加关键数据日志记录点，跑一轮训练（少量 iteration），记录基线数据：

#### 日志记录点（临时添加，用于回归对比）

1. **数据加载后**（`train.py:218` 之后）：
   ```python
   # 记录第1个batch的图像/标签统计
   logging.info(f"[BASELINE] image shape={image_batch.shape}, dtype={image_batch.dtype}, "
                f"min={image_batch.min():.4f}, max={image_batch.max():.4f}, mean={image_batch.mean():.4f}")
   logging.info(f"[BASELINE] label shape={label_batch.shape}, unique={torch.unique(label_batch).tolist()}")
   ```

2. **模型输出后**（`train.py:227` 之后）：
   ```python
   logging.info(f"[BASELINE] output shape={outputs.shape}, softmax range=[{outputs_soft.min():.4f}, {outputs_soft.max():.4f}]")
   ```

3. **Loss 计算后**（`train.py:231` 之后）：
   ```python
   logging.info(f"[BASELINE] loss={loss.item():.6f}, ce={ce_loss(outputs, label_batch.long()).item():.6f}, "
                f"dice={losses.dice_loss(outputs_soft[:, 1, ...], label_batch).item():.6f}")
   ```

4. **验证指标**（`train.py:259` 之后）：
   ```python
   logging.info(f"[BASELINE] val iter={iter_num}, dice={np.mean(metric_list):.6f}, iou={np.mean(metric_list2):.6f}")
   ```

5. **数据集信息**（`train.py:187` 之后）：
   ```python
   logging.info(f"[BASELINE] dataset: train={len(db_train)}, val={len(db_val)}")
   sample = db_train[0]
   logging.info(f"[BASELINE] sample keys={list(sample.keys())}, image={sample['image'].shape}, label={sample['label'].shape}")
   ```

#### 执行基线采集

```bash
# 修改前代码，跑 ~50 iterations（足够采集基线数据）
python train.py --root_path <PRIME-FP20路径> --max_iterations 50
# 复制 log.txt 为 baseline_log.txt
cp <snapshot_path>/log.txt baseline_log.txt
```

### 阶段二：实施修改

按照本计划修改代码，新建文件。

### 阶段三：修改后回归验证

用 `--dataset prime_fp20` 配置运行，**保留同样的日志记录点**：

```bash
# 修改后代码，同样跑 ~50 iterations
python train.py --dataset prime_fp20 --max_iterations 50
```

#### 对比检查项

| 检查项 | 对比方法 | 预期结果 |
|--------|----------|----------|
| 数据集样本数 | `[BASELINE] dataset: train=N, val=N` | 与基线一致 |
| 样本格式 | `[BASELINE] sample keys/image/label shape` | 与基线一致 |
| 图像统计 | `image shape/min/max/mean` | 与基线一致（允许浮点误差 <1e-4） |
| 标签值 | `label unique values` | `[0, 1]` 二值，与基线一致 |
| 模型输出 | `output shape/softmax range` | 与基线一致 |
| Loss 值 | `loss/ce/dice` 前10个iter | 差异 <1e-3（随机性导致的微小差异可接受） |
| 验证指标 | `val dice/iou` | 与基线一致 |

**关键判定标准**：如果相同 seed、相同数据、相同超参下，前 10 个 iteration 的 loss 值与基线逐位一致，则说明数据加载和模型行为完全一致。

### 阶段四：扩展功能验证

回归通过后，验证新数据集支持：

```bash
# 用 fundus_dataset 测试新数据集加载
python train.py --dataset fundus --max_iterations 10
# 确认：图像加载正常、transform 正常、loss 下降
```

---

## 7. 不做的事

- 不抽取 `unet_config` 到配置文件（模型架构参数，超出数据集扩展范围）
- 不修改 `BaseDataSets`（保留向后兼容）
- 不修改模型代码 `SerpMamba.py`
- 不引入额外的抽象层/工厂模式/注册表
- 不添加不必要的异常处理和兜底逻辑
- 不做可视化扩展（后续单独做）
- 不做零配置自动推断（仅 YAML 配置模式）

---

## 8. 模型测试操作流程

### 前置条件

训练完成后，模型权重保存在 `experiments/<数据集名>/` 目录下，包含以下文件：

```
experiments/fundus/
├── unet_best_model.pth          # 最佳模型（val Dice 最高的 checkpoint）
├── model_iter_2000.pth          # 每 2000 步的定期 checkpoint
├── model_iter_4000.pth
├── ...
├── model_iter_15200_dice_0.8234.pth  # 达到 best 时的 checkpoint
├── tf-logs/                     # TensorBoard 日志
├── training_stats.csv           # 训练统计数据
├── training_plots/              # 训练曲线图
└── training_report.txt          # 训练汇总报告
```

### 测试命令

```bash
cd Serp_Mamba

# 默认：只测试 best model（*_best_model.pth）
python test.py --dataset fundus

# 测试所有 checkpoint（遍历目录下全部 .pth）
python test.py --dataset fundus --test_all

# 指定 checkpoint 目录
python test.py --dataset fundus --checkpoint_dir experiments/fundus

# 指定测试集 split（默认 "test"，也可用 "val" 做验证集评估）
python test.py --dataset fundus --split test

# 自定义输出目录（默认保存到 <checkpoint_dir>/test_results/）
python test.py --dataset fundus --save_dir results/fundus_test
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | `prime_fp20` | 数据集配置名（对应 `configs/<name>.yaml`）或 YAML 文件路径 |
| `--checkpoint_dir` | `experiments/<数据集名>` | 存放 .pth 权重文件的目录 |
| `--split` | `test` | 测试的数据集划分：`test`、`val` 或 `train` |
| `--test_all` | `False` | 加上此 flag 则遍历目录下所有 .pth；否则只测 `*_best_model.pth` |
| `--num_classes` | `2` | 输出类别数 |
| `--save_dir` | `<checkpoint_dir>/test_results` | 输出目录 |
| `--root_path` | (配置文件中的 `root_dir`) | 可覆盖配置中的数据集路径 |

### 测试输出

测试完成后，在 `save_dir`（默认 `<checkpoint_dir>/test_results/`）下生成：

```
test_results/
├── predictions/                 # 逐张预测掩码（白色=血管，黑色=背景）
│   ├── image001.png
│   ├── image002.png
│   └── ...
└── output.txt                   # 汇总指标 + 逐图指标
```

`output.txt` 内容示例：

```
model_name = experiments/fundus/unet_best_model.pth
split = test
num_samples = 20
Dice = mean-sd = 0.8234-0.0456
Iou = mean-sd = 0.7012-0.0523
MCC = mean-sd = 0.7891-0.0412
BM = mean-sd = 0.7523-0.0389

per-image results:
  image001.jpg: dice=0.8512, iou=0.7401, mcc=0.8123, bm=0.7845
  image002.jpg: dice=0.7956, iou=0.6623, mcc=0.7654, bm=0.7201
  ...
```

### 测试的 4 个指标

| 指标 | 含义 | 理想值 |
|------|------|--------|
| Dice | 分割重叠度 | 越接近 1 越好 |
| IoU | 交并比（Jaccard） | 越接近 1 越好 |
| MCC | Matthews 相关系数 | 越接近 1 越好 |
| BM | Bookmaker Informedness（真正率+真负率-1） | 越接近 1 越好 |

### 典型工作流

```bash
# 1. 训练
python train.py --dataset fundus --max_iterations 20000

# 2. 查看训练报告，确认 best model 性能
cat experiments/fundus/training_report.txt

# 3. 测试（默认只测 best model）
python test.py --dataset fundus
python test.py --dataset prime_fp20

# 4. 如需对比所有 checkpoint 的性能差异
python test.py --dataset fundus --test_all
```

---

## 9. 新增数据集实例：DRIVE

> 时间戳：2026-05-25

### 9.1 数据集概况

| 属性 | 值 |
|------|-----|
| 名称 | DRIVE (Digital Retinal Images for Vessel Extraction) |
| 图像分辨率 | 565 × 584 |
| 图像格式 | TIFF (RGB 彩色) |
| 标签格式 | GIF (灰度二值 0/255) |
| 训练集 | 20 张 (编号 21-40) |
| 测试集 | 20 张 (编号 01-20) |
| 标注 | `1st_manual/`（第一观察者），`2nd_manual/` 忽略 |
| FOV 掩膜 | `mask/`（当前流程未使用） |

### 9.2 目录结构

```
datasets/DRIVE/
├── training/
│   ├── images/          # 21_training.tif ~ 40_training.tif
│   ├── 1st_manual/      # 21_manual1.gif ~ 40_manual1.gif
│   └── mask/            # 21_training_mask.gif ~ 40_training_mask.gif
└── test/
    ├── images/          # 01_test.tif ~ 20_test.tif
    ├── 1st_manual/      # 01_manual1.gif ~ 20_manual1.gif
    ├── 2nd_manual/      # (忽略)
    └── mask/            # 01_test_mask.gif ~ 20_test_mask.gif
```

### 9.3 实施内容

**仅新增 `configs/drive.yaml`，零代码修改。**

配置要点：
- `label_name_transform`：两条 replace 规则分别处理 `_training` → `_manual1` 和 `_test` → `_manual1`，再加上 `change_ext: ".gif"`
- `image_mode: "RGB"` + `input_channels: 1`：RGB 图像转灰度送入模型
- `split_map`：DRIVE 无独立 val 划分，`val` 暂映射到 `test`
- `patch_size: [512, 512]`：DRIVE 图像较小，使用 512 patch

### 9.4 同步更新

所有数据集已统一存放至 `datasets/` 目录，三个配置的 `root_dir` 路径已同步更新为相对路径：

| 配置文件 | 旧路径 | 新路径 |
|----------|--------|--------|
| `prime_fp20.yaml` | `/home/lishh237/Serp-Mamba/Serp_Mamba/PRIME-FP20_DataPort/...` | `datasets/PRIME-FP20_DataPort/...` |
| `fundus.yaml` | `U_Mamba_main/data/fundus_dataset` | `datasets/fundus_dataset` |
| `drive.yaml` | (新建) | `datasets/DRIVE` |

### 9.5 DRIVE 工作流

```bash
# 训练
python train.py --dataset drive --max_iterations 20000

# 测试 best model
python test.py --dataset drive

# 查看训练报告
cat experiments/DRIVE/training_report.txt
```

### 9.6 注意事项

- **无独立 val 集**：DRIVE 标准做法是训练集 20 张训练、测试集 20 张评估，无中间验证。当前配置将 `val` 映射到 `test`，训练过程中的 val 指标即测试指标。
- **FOV 掩膜**：DRIVE 提供 `mask/` 视野掩膜用于限定评估区域。当前流程未使用 FOV 掩膜，指标在全图计算。如需严格按 DRIVE 论文标准评估（仅 FOV 内计算指标），需后续扩展。
- **图像尺寸**：DRIVE 分辨率 (565×584) 远小于 PRIME-FP20，`patch_size: 512` 已接近全图尺寸，数据增强的随机裁剪效果有限。

---

## 10. 无标签数据集推理支持

> 时间戳：2026-05-26

### 10.1 背景

部分数据集（如 FIRE）只有原始眼底图像，没有分割标注。用户希望用已有模型（如 DRIVE 训练出的模型）对这些数据集进行跨数据集推理，仅输出分割掩码，不计算指标。

### 10.2 实施内容

**修改 2 个文件，约 25 行改动，不破坏已有功能：**

#### `dataloaders/dataset_registry.py`

- `ConfigDataSets.__init__`：新增 `self.has_labels = config.get("has_labels", True)`
- `ConfigDataSets.__getitem__`：`has_labels=False` 时跳过 `_load_label()`，返回与图像同尺寸的零填充 dummy label

#### `test.py`

- `test_single_volume_fast`：新增 `has_label` 参数，`False` 时保存预测掩码后直接返回空指标
- `Inference` 函数：从配置读取 `has_labels`，传递给测试函数；无标签时跳过指标聚合，`output.txt` 仅记录模型信息和预测文件列表

### 10.3 设计依据

- **为什么返回 dummy label 而非 None**：下游 `DataLoader` 的 collate 机制要求 batch 内所有样本的 tensor 形状一致。返回与图像同尺寸的零数组，既满足 collate 约束，又不触发 metric 计算（被 `has_label` 条件跳过）。
- **为什么在 config 层而非命令行控制**：一个数据集是否有标签是数据集的固有属性，不是运行时选择。用 YAML 的 `has_labels` 字段表达，与 `image_mode`、`input_channels` 等属性一致。

### 10.4 实例：FIRE 数据集配置

FIRE (Fundus Image Registration Dataset) 的 Ground Truth 为配准控制点（非分割标签），仅用于无标签推理。

#### 数据集概况

| 属性 | 值 |
|------|-----|
| 名称 | FIRE |
| 图像分辨率 | 2912 × 2912 |
| 图像格式 | JPG (RGB 彩色) |
| 标注 | 配准控制点（`.txt`），非分割标签 |
| 分类 | A (28), A-Robotic (110), B-Manual (160), P (98), S (142) |

#### 目录结构

```
datasets/dataset_fire/
├── A/                          # 28 张配对图像
│   ├── Images/                 # A01_1.jpg, A01_2.jpg, ...
│   └── Ground Truth/           # control_points_*.txt（配准标签，忽略）
├── A-Robotic/                  # 110 张
│   ├── Images/
│   └── Ground Truth/
├── B-Manual/                   # 160 张
│   ├── Images/
│   └── Ground Truth/
├── P/                          # 98 张
│   ├── Images/
│   └── Ground Truth/
└── S/                          # 142 张
    ├── Images/
    └── Ground Truth/
```

#### 配置要点

- `has_labels: false`：无分割标签，跳过标签加载和 `label_dir` 必填校验
- `image_dir: "{split}/Images"`：`{split}` 对应分类目录名（A, A-Robotic 等）
- `split_map` 中 `test: "A"` 为默认分类；其他分类不在 map 中，通过 `--split` 直接指定目录名
- `patch_size: [1024, 1024]`：FIRE 图像分辨率高 (2912×2912)

### 10.5 FIRE 推理命令

```bash
# 用 DRIVE 训练的模型预测 FIRE-A 分类（默认，28 张）
python test.py --dataset fire --checkpoint_dir experiments/DRIVE

# 预测其他分类
python test.py --dataset fire --split A-Robotic --checkpoint_dir experiments/DRIVE
python test.py --dataset fire --split B-Manual --checkpoint_dir experiments/DRIVE
python test.py --dataset fire --split P --checkpoint_dir experiments/DRIVE
python test.py --dataset fire --split S --checkpoint_dir experiments/DRIVE

# 指定输出目录
python test.py --dataset fire --checkpoint_dir experiments/DRIVE --save_dir results/fire_A

# 遍历所有 checkpoint 生成预测
python test.py --dataset fire --split A-Robotic --checkpoint_dir experiments/DRIVE --test_all

# 用其他数据集训练的模型预测
python test.py --dataset fire --checkpoint_dir experiments/fundus
python test.py --dataset fire --checkpoint_dir experiments/PRIME-FP20
```

### 10.6 无标签模式输出

```
test_results/
├── predictions/          # 分割掩码（白色=血管，黑色=背景）
│   ├── A01_1.png
│   ├── A01_2.png
│   └── ...
└── output.txt           # 仅含模型信息，无指标
```

`output.txt` 内容：

```
model_name = experiments/DRIVE/unet_best_model.pth
split = A
num_samples = 28
note = no ground truth labels, predictions only
```
