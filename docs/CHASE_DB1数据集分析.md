# CHASE_DB1 数据集核心认知与科研使用手册
本文档基于 CHASE_DB1 原始论文（Fraz et al., 2012, IEEE TBME）及 Owen et al., 2011（Arterioscler Thromb Vasc Biol — CHASE 儿童心血管研究），无外推、无自创，用于：
- 精准定位**儿童视网膜血管分割**场景
- 规范使用数据、标签、评价指标
- 科学分析模型不足与改进方向
- 可直接作为论文"Dataset"章节使用

---

## 一、数据集定位与目标场景
### 1. 数据集全称与来源
- **名称**：CHASE_DB1（Child Heart and Health Study in England — Database 1）
- **发表**：Fraz et al., 2012, IEEE Transactions on Biomedical Engineering
- **原始数据来源**：Owen et al., 2011, Arterioscler Thromb Vasc Biol — CHASE 儿童心血管健康研究
- **注意**：Owen 2011 是 CHASE 原始数据采集论文（986 名儿童，研究血管弯曲度与心血管风险），并非 CHASE_DB1 血管分割数据集论文；CHASE_DB1 是 Fraz 2012 从中抽取 28 张并标注血管分割的结果
- **核心贡献**：提供**双标注者独立标注**的儿童视网膜血管分割数据，广泛用于血管分割算法基准测试

### 2. 官方明确的目标场景
1. **彩色眼底照片视网膜血管自动分割**
2. **血管形态定量分析（弯曲度、直径等）**
3. **儿童心血管健康筛查**
4. **多标注者一致性研究**

### 3. 与临床/科研场景匹配
- 儿童眼底血管形态 → 心血管风险早期预测
- 双标注者 → 标注不确定性研究
- 小样本高标注质量 → 少样本学习、跨域迁移

---

## 二、数据集结构与文件含义
### 1. 样本与规模
- 总样本：**28 张彩色眼底图**（14 名学童，每人左右眼各 1 张）
- 受试者：伦敦地区 10 岁儿童，南亚和欧洲裔
- 病种：**正常健康儿童**（无病种分类）
- **注意**：这是所有主流血管分割数据集中**样本量最小**的

### 2. 数据格式与分辨率
- 设备：Nidek NM-200D 手持眼底相机，30° 视野（Owen 2011 原始采集为 1280×960，公开发布版本为 999×960）
- 分辨率：**999 × 960 px**，JPG 格式
- 标注：**像素级二值血管掩码**（白=血管，黑=背景），PNG 格式
- **双标注者**：每张图像提供两个独立标注（_1stHO 和 _2ndHO）

### 3. 文件组织结构
```
CHASE_DB1/
├── raw/                          # 原始数据（本项目的实际使用目录）
│   ├── Image_01L.jpg             # 第1名儿童左眼
│   ├── Image_01R.jpg             # 第1名儿童右眼
│   ├── ...
│   ├── Image_14L.jpg             # 第14名儿童左眼
│   ├── Image_14R.jpg             # 第14名儿童右眼
│   ├── Image_01L_1stHO.png       # 第1名儿童左眼 - 第1标注者掩码
│   ├── Image_01L_2ndHO.png       # 第1名儿童左眼 - 第2标注者掩码
│   └── ...
├── dsdl/                         # DSDL 标准格式（OpenDataLab 标准）
├── sample/                       # 预览缩略图
├── README.md
└── metafile.yaml
```

### 4. 命名规则
- 原图：`Image_{编号}{L/R}.jpg`（L=左眼，R=右眼）
- 标注：`Image_{编号}{L/R}_{1stHO/2ndHO}.png`
- 示例：`Image_03R_1stHO.png` → 第3名儿童右眼，第1标注者掩码

---

## 三、双标注者说明
CHASE_DB1 最显著的特点是**每张图像提供两个独立标注**：

| 标注 | 含义 | 用法 |
|------|------|------|
| `_1stHO` | 第 1 人类观察者（1st Human Observer） | 通常作为训练/评估的**标准标签** |
| `_2ndHO` | 第 2 人类观察者（2nd Human Observer） | 可用于：标注一致性分析、鲁棒性评估、集成训练 |

### 常见用法
1. **标准用法**：仅使用 `_1stHO` 作为 GT，与文献基准一致
2. **双 GT 评估**：分别对两个标注计算指标，报告范围
3. **集成训练**：两个标注轮流或加权使用，增强模型鲁棒性
4. **不确定性研究**：两个标注的差异区域作为标注不确定性地图

---

## 四、标注质量（原文关键信息）
- 标注者：两名专家独立完成
- 原始研究目标：**血管弯曲度测量**（而非纯分割），因此对细血管、边界精度要求高
- 标注差异主要存在于：细血管判别、血管边界像素、视盘边缘区域

---

## 五、如何正确使用数据（训练/测试/评测）
### 1. 训练/测试划分（文献常见做法）
CHASE_DB1 **没有官方训练/测试划分**。文献中常见的做法：

| 划分方式 | 训练集 | 测试集 | 说明 |
|----------|--------|--------|------|
| 随机 21/7 | 随机选 21 张 | 剩余 7 张 | **本项目采用**。Chen et al. 2026 (MDCT-Unet) 参考 Zhou et al. 2021 (ICCV) 的做法，随机划分 75%/25% |
| 固定前20/后8 | 前 20 张 (01–10 L/R) | 后 8 张 (11–14 L/R) | 部分文献使用，但并非原文定义 |
| k-fold 交叉验证 | — | — | 适合小样本严格评估 |

**注意**：
- 本项目采用 21/7 随机划分（val_split_ratio=0.25），通过固定种子保证可复现
- Fraz 2012 原文使用像素级分类策略，未固定训练/测试集划分

### 2. 输入处理
- 推荐使用**绿色通道或灰度**（血管对比度最高）
- 图像分辨率较低（999×960），无需复杂预处理
- patch_size 建议：512×512 或 256×256

### 3. 典型错误用法
- 不要将 `_2ndHO` 混入 `_1stHO` 作为同一 GT（标注有差异）
- 不要仅用 28 张做训练而不做数据增强（严重过拟合）
- 不要忽略数据量极小对评价指标方差的影响

---

## 六、本项目中的使用方式（YAML 配置驱动）
本项目通过 `Serp_Mamba/configs/` 下的 YAML 配置文件加载数据集。需创建 `chase_db1.yaml` 配置文件：

```yaml
# CHASE_DB1 视网膜血管分割数据集配置
name: "CHASE_DB1"
description: "CHASE_DB1: Child Heart and Health Study (Fraz et al., 2012)"

root_dir: "datasets/CHASE_DB1/raw"

# CHASE_DB1 无 train/test 子目录，所有数据平铺在 raw/ 下
# 通过 val_split_ratio 从训练集划出验证集
image_dir: "."
label_dir: "."

split_map:
  train: ""
  val: ""
  test: ""

# 文件匹配
image_ext: ".jpg"

# 标签文件名推导：Image_01L.jpg → Image_01L_1stHO.png
label_name_transform:
  - append_suffix: "_1stHO"
  - change_ext: ".png"

# 图像属性
image_mode: "RGB"
input_channels: 1          # 转灰度送入模型
label_threshold: 0         # 二值掩码，>0 即为血管
label_rgb_mode: "mean"

# 验证集划分（28 张中划出 25% 即 7 张作为验证集，与 Chen 2026 一致）
val_split_ratio: 0.25
val_split_seed: 1337

# 训练默认超参
# Chen 2026 (MDCT-Unet) 使用 batch_size=12, 960×960, 300 epochs
# 但 SerpMamba 的 Mamba SSM 层显存消耗远大于 MDCT-Unet，
# 需降低 patch_size 和 batch_size 以适配 V100 32GB
# 等效训练量：300 epochs × 21 samples/epoch = 6300 iterations
patch_size: [512, 512]
batch_size: 1
max_iterations: 6300
base_lr: 0.0001
num_classes: 2
```

训练命令：
```bash
python train.py --dataset chase_db1 --root_path <实际路径>
```

---

## 七、官方推荐评价指标
### 1. 基础分割指标
- **Dice / F1-Score**（核心）
- **Sensitivity (Recall / SE)**：血管检出率
- **Specificity (SP)**：背景准确率
- **Precision (PR)**：预测精确率
- **Accuracy (ACC)**
- **AUC**
- **IoU (Jaccard)**

### 2. 特殊考量
- **对两个 GT 分别评估**，报告范围（如 Dice: 0.78–0.82）
- 小样本下指标方差大，建议用交叉验证
- 应报告**细血管/粗血管分层指标**

---

## 八、与其他血管分割数据集对比

| 特征 | CHASE_DB1 | DRIVE | STARE | FIVES | HRF |
|------|-----------|-------|-------|-------|-----|
| 样本数 | **28** | 40 | 20 | **800** | 45 |
| 分辨率 | 999×960 | 565×584 | 700×605 | 2048×2048 | 3504×2336 |
| 标注者 | **2** | 1(+1) | 2 | 多级 | 2 |
| 受试者 | 儿童 | 成人 | 多年龄 | 多病种 | 健康+病 |
| FOV | ~30° | 45° | 35° | 50° | 60° |
| 数据量级别 | 极小 | 小 | 极小 | 大 | 中 |

---

## 九、模型不足分析（基于 CHASE_DB1 可直接验证）
### 1. 小样本过拟合
- 仅 28 张图，模型极易过拟合
- 需要强数据增强、正则化、预训练迁移

### 2. 分辨率与细节限制
- 999×960 分辨率下细血管仅 1–2 px
- 分割结果难以精确测量血管宽度

### 3. 双标注差异
- 两位标注者在细血管区域差异显著
- 模型可能学到一个标注者偏向，对另一个泛化差

### 4. 仅含正常儿童数据
- 无法验证模型在病变、成人、不同族群上的泛化性

---

## 十、面向临床实用的改进方向
### 1. 小样本学习
- 元学习、少样本分割
- 跨数据集预训练（如先在 FIVES 上预训练，再迁移到 CHASE_DB1）

### 2. 标注不确定性建模
- 利用双标注训练不确定性感知模型
- 对标注不一致区域降低损失权重

### 3. 跨域泛化
- 与 DRIVE/FIVES 等联合训练，增强跨域鲁棒性
- 域自适应技术

---

## 十一、prepare.py 的作用说明

`Serp_Mamba/datasets/CHASE_DB1/dsdl/dsdl_SemSeg_full/tools/prepare.py` 是 **DSDL（Data Set Description Language）格式转换工具**，由 OpenDataLab 提供，作用如下：

### 功能
1. **解压原始数据**（如 zip/tar.gz）到 prepared 目录
2. **扫描 raw 目录**中的 `.jpg` 图像文件
3. **自动匹配对应的标注文件**（`_1stHO.png` 和 `_2ndHO.png`）
4. **生成 DSDL 标准格式的标注文件**：
   - `train_samples.json`：每条记录包含 image、label_map_1stHO、label_map_2ndHO
   - `train.yaml`：DSDL 子集描述
   - `class-dom.yaml`：类别定义（retinal 类）
   - `config.py`：数据读取配置

### 使用方法
```bash
# 数据已解压，直接转换
python tools/prepare.py -d <raw_data_path>

# 从压缩包开始
python tools/prepare.py <compressed_data_path>
```

### 与本项目的关系
**本项目不使用 DSDL 格式加载 CHASE_DB1。** 本项目通过 `dataloaders/dataset_registry.py` 中的 `ConfigDataSets` 类 + YAML 配置直接读取 `raw/` 目录下的图像和掩码文件。`dsdl/` 目录和 `prepare.py` 仅作为 OpenDataLab 下载数据时附带的标准格式参考，可以忽略。

如果希望使用 DSDL 格式，需要安装 `dsdl` SDK 并在 `config.py` 中配置路径。

---

## 十二、可直接写入论文的标准表述
CHASE_DB1（Child Heart and Health Study in England）是一个公开的视网膜血管分割基准数据集，包含 28 张 999×960 像素的彩色眼底图像，采集自 14 名 10 岁儿童的双眼。每张图像由两名独立标注者提供像素级血管分割掩码，标注差异主要集中于细血管区域。该数据集样本量较小，适用于少样本学习、标注不确定性分析以及跨域迁移等研究场景。评估时建议对两个标注分别计算 Dice、Sensitivity、Specificity 等指标并报告范围，以充分反映标注差异对模型性能评估的影响。

---

## 引文
```bibtex
@article{fraz2012ensemble,
  title={An ensemble classification-based approach applied to retinal blood vessel segmentation},
  author={Fraz, Muhammad Moazam and Remagnino, Paolo and Hoppe, Andreas and Uyyanonvara, Bunyarit and Rudnicka, Alicja R and Owen, Christopher G and Barman, Sarah A},
  journal={IEEE Transactions on Biomedical Engineering},
  volume={59},
  number={9},
  pages={2538--2548},
  year={2012},
  publisher={IEEE}
}

@article{owen2011retinal,
  title={Retinal Arteriolar Tortuosity and Cardiovascular Risk Factors in a Multi-Ethnic Population Study of 10-Year-Old Children; the Child Heart and Health Study in England (CHASE)},
  author={Owen, Christopher G and Rudnicka, Alicja R and Nightingale, Claire M and Mullen, Robert and Barman, Sarah A and Sattar, Naveed and Cook, Derek G and Whincup, Peter H},
  journal={Arteriosclerosis, Thrombosis, and Vascular Biology},
  volume={31},
  number={8},
  pages={1933--1938},
  year={2011},
  publisher={Lippincott Williams \& Wilkins}
}
```
