# Serp-Mamba 模型导出与推理集成方案

> 时间戳：2026-05-26

## Context

基于 fundus 数据集训练的 Serp-Mamba 模型（`experiments/fundus/unet_best_model.pth`），需要在其他项目中进行血管分割推理。

### 核心约束

1. **SerpMamba 依赖 `mamba_ssm` + `causal_conv1d`（自定义 CUDA 内核），无法导出为 ONNX/TorchScript**
2. **推理必须使用 CUDA GPU**，`mamba_ssm` 不兼容 CPU

### 需求

1. **快速同步**：Serp-Mamba 项目持续迭代，改动应能快速反映到使用方，不需要两边重复修改
2. **封装隔离**：使用方只需关心输入图片、输出掩码，不暴露模型构造细节
3. **自包含可迁移**：封装为独立可安装的 pip 包，特定依赖（mamba_ssm 等）由包管理，使用方无需关心

---

## 1. 方案：pip 可安装推理包

在 Serp-Mamba 仓库内创建 `serpmamba/` pip 包。使用方通过 `pip install -e` 安装后，直接调用 `load_model()` + `predict()` 即可。

### 1.1 仓库目录结构

```
Serp_Mamba/
├── serpmamba/                            # pip 可安装的推理包
│   ├── pyproject.toml                    # 包元信息 + 依赖声明
│   ├── install.sh                        # 一键安装（先装 CUDA wheels，再装包）
│   ├── serpmamba/
│   │   ├── __init__.py                   # 公共 API：load_model, predict
│   │   ├── model.py                      # 模型构造 + 权重加载
│   │   ├── predict.py                    # 预处理 + 推理 + 后处理
│   │   ├── _model_def/                   # 模型定义（从原始位置提取）
│   │   │   ├── __init__.py               # from .SerpMamba import SerpMamba
│   │   │   ├── SerpMamba.py              # 原始文件，去除 nnunetv2 依赖
│   │   │   └── requirements.txt          # 模型定义的特定依赖
│   │   └── wheels/                       # 预编译的 CUDA 扩展
│   │       ├── mamba_ssm-1.1.1+cu122...whl
│   │       └── causal_conv1d-1.1.3+cu122...whl
│   └── README.md                         # 安装和使用说明
├── experiments/
│   └── fundus/
│       └── unet_best_model.pth           # 训练好的权重
├── U_Mamba_main/umamba/nnunetv2/nets/
│   └── SerpMamba.py                      # 原始模型定义（开发用）
└── ...
```

### 1.2 快速同步策略

`_model_def/SerpMamba.py` 是从原始 `U_Mamba_main/.../SerpMamba.py` 提取的独立版本（去除 nnunetv2 依赖）。同步方式：

- **开发期间**：模型架构基本冻结，训练出的 .pth 权重与架构绑定，不频繁变动
- **架构更新时**：重新运行提取步骤（一次性脚本，去除 nnunetv2 的两个 import 并将相关函数改为延迟导入）
- **权重更新时**：只需替换 `.pth` 文件路径，无需重装包

### 1.3 nnunetv2 依赖消除

原始 `SerpMamba.py` 顶层 import 了两个 nnunetv2 模块，但仅在一个工厂函数 `get_umamba_enc_from_plans()` 中使用（推理时不调用）。处理方式：

```python
# 原始代码（顶层 import，强制依赖）
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
from nnunetv2.utilities.network_initialization import InitWeights_He

# 改为延迟导入（仅在工厂函数内部 import，推理路径不触发）
def get_umamba_enc_from_plans(...):
    from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager, PlansManager
    from nnunetv2.utilities.network_initialization import InitWeights_He
    ...
```

改动后推理包不再依赖 nnunetv2 项目目录结构。

---

## 2. 公共 API

### 2.1 `load_model()`

```python
from serpmamba import load_model

net = load_model("path/to/unet_best_model.pth")
# 返回已加载权重、处于 eval 模式、在 CUDA 上的模型
```

### 2.2 `predict()`

```python
from serpmamba import predict

# 方式一：文件路径输入
mask = predict(net, "retina.jpg")
# → numpy 数组 (H, W)，值域 {0, 1}，0=背景, 1=血管

# 方式二：PIL Image 输入
from PIL import Image
img = Image.open("retina.jpg")
mask = predict(net, img)

# 方式三：保存到文件
mask = predict(net, "retina.jpg", save_path="mask.png")
```

### 2.3 `__init__.py`

```python
from serpmamba.model import load_model
from serpmamba.predict import predict

__all__ = ["load_model", "predict"]
```

---

## 3. 依赖管理

### 3.1 `pyproject.toml`

```toml
[project]
name = "serpmamba"
version = "0.1.0"
description = "SerpMamba retinal vessel segmentation inference"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.1",
    "einops>=0.8",
    "dynamic_network_architectures>=0.3",
    "numpy",
    "Pillow",
    "scipy",
]
# mamba_ssm 和 causal_conv1d 是编译好的 CUDA wheel，
# 需先通过 wheels/ 安装（见 install.sh）

[project.optional-dependencies]
dev = ["mamba_ssm", "causal_conv1d"]
```

### 3.2 `install.sh`

```bash
#!/bin/bash
# 安装预编译的 CUDA 扩展（必须匹配 CUDA 12.2 + PyTorch 2.1 + Python 3.10）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install "$SCRIPT_DIR/serpmamba/wheels/mamba_ssm-"*.whl
pip install "$SCRIPT_DIR/serpmamba/wheels/causal_conv1d-"*.whl

# 安装推理包（可编辑模式，便于同步更新）
pip install -e "$SCRIPT_DIR"
```

### 3.3 使用方视角

使用方只需执行两步：

```bash
# 1. 一键安装（CUDA wheels + 包 + 所有依赖）
cd /path/to/Serp-Mamba/Serp_Mamba/serpmamba
bash install.sh

# 2. 使用
python -c "from serpmamba import load_model, predict; \
           net = load_model('../experiments/fundus/unet_best_model.pth'); \
           mask = predict(net, 'test.jpg', save_path='mask.png')"
```

---

## 4. 内部实现

### 4.1 `model.py`

```python
import torch
import torch.nn as nn
from serpmamba._model_def import SerpMamba


def load_model(weights_path, device="cuda"):
    """加载模型权重，返回处于 eval 模式的模型。"""
    net = SerpMamba(
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        conv_op=nn.Conv2d,
        kernel_sizes=[[3, 3]] * 6,
        strides=[[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        n_conv_per_stage_decoder=[2, 2, 2, 2, 2],
        num_classes=2,
        conv_bias=True,
        norm_op=nn.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        dropout_op=None,
        dropout_op_kwargs=None,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    net.load_state_dict(torch.load(weights_path, map_location=device)["state_dict"])
    net.to(device).eval()
    return net
```

### 4.2 `predict.py`

```python
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import zoom

PATCH_SIZE = (1024, 1024)


def predict(net, image, save_path=None, device="cuda"):
    """对单张图像进行血管分割推理。

    参数:
        net: load_model() 返回的模型
        image: 文件路径 (str) 或 PIL.Image 对象
        save_path: 可选，保存预测掩码为 PNG
        device: 推理设备

    返回:
        numpy 数组 (H, W)，值域 {0, 1}
    """
    if isinstance(image, str):
        image = Image.open(image)
    img = image.convert("L")
    img_array = np.array(img)
    h, w = img_array.shape

    resized = zoom(img_array, (PATCH_SIZE[0] / h, PATCH_SIZE[1] / w), order=0)
    tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = net(tensor)
        pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
        pred = pred.cpu().numpy().squeeze()

    mask = zoom(pred, (h / PATCH_SIZE[0], w / PATCH_SIZE[1]), order=0).astype(np.uint8)

    if save_path:
        Image.fromarray(mask * 255).save(save_path)

    return mask
```

---

## 5. 在其他项目中集成

### 5.1 方式一：本地路径安装（推荐，支持快速同步）

```bash
# 在目标项目中
pip install -e /path/to/Serp-Mamba/Serp_Mamba/serpmamba
```

Serp-Mamba 仓库更新后（git pull），无需重装，改动立即生效。

### 5.2 方式二：git 仓库安装

```bash
# 从 git 仓库直接安装
pip install -e "git+https://github.com/xxx/Serp-Mamba.git#subdirectory=Serp_Mamba/serpmamba"
```

### 5.3 在目标项目代码中使用

```python
from serpmamba import load_model, predict

# 初始化（通常在服务启动时调用一次）
net = load_model("path/to/unet_best_model.pth")

# 推理（每次请求调用）
mask = predict(net, "input_image.jpg")
# mask 是 numpy 数组 (H, W)，值为 0 或 1
```

使用方只需关心 3 点：
1. 安装包：`pip install -e ...`
2. 模型权重：`.pth` 文件路径
3. 调用：`load_model()` + `predict()`

无需了解 SerpMamba 架构、模型参数、预处理流程、CUDA 依赖细节。

---

## 6. 注意事项

### 6.1 环境兼容性

`mamba_ssm` + `causal_conv1d` 是预编译 CUDA 扩展，要求：
- PyTorch 2.1.x
- CUDA 12.2
- Python 3.10
- NVIDIA GPU

目标项目环境必须满足。如需其他 CUDA/PyTorch 版本，需重新编译 mamba_ssm。

### 6.2 CPU 推理

**不可行**。`mamba_ssm` 的核心算子只有 CUDA 实现。如需 CPU 推理，必须将 Mamba 层替换为纯 PyTorch 等效实现（超出当前范围）。

### 6.3 输入格式

- 任意分辨率均可，自动 resize 到 1024×1024 再还原
- RGB 彩色图会自动转灰度
- 输出与输入尺寸一致

---

## 7. 不做的事

- **不导出 ONNX / TorchScript**：`mamba_ssm` 自定义 CUDA 内核无法被 tracer 追踪
- **不封装为 REST API**：使用方可基于此包自行用 FastAPI/Flask 包装
- **不做模型训练接口暴露**：推理包只负责推理，训练仍在 Serp-Mamba 项目内进行
- **不做多模型管理**：当前只支持单个 .pth 加载，如需多模型切换由使用方自行管理
