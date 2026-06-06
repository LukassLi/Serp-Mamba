# SerpMamba 改进方案清单

> 基于 FIVES 实验分析（2026-06-06 修正版）暴露的核心问题（Glaucoma 细血管丢失、DR 假阳性、HD95 偏高、训练效率低），筛选**有开源代码、可直接集成、性价比高**的方法。
>
> 与 `experiment_analysis.md` 的七、八节配套阅读。

---

## 第一梯队：立即可集成（仅需修改训练脚本）

### 1. CLAHE 预处理（预计 Dice +2~3%）

- **来源**：berenslab 基准论文（arXiv:2406.14994）的标准预处理流程
- **原理**：限制对比度自适应直方图均衡化，增强血管与背景的局部对比度
- **实现**：`skimage.exposure.equalize_adapthist(img, clip_limit=2.0, kernel_size=(8,8))`
- **预期效果**：直接提升低对比度图像的分割质量
- **集成方式**：在 `dataloaders/dataset_registry.py` 的 `_load_image()` 中添加 1 行预处理
- **集成成本**：极低
- **优先级**：⭐⭐⭐⭐⭐
- **状态**：待实施

### 2. clDice 损失函数（预计 Dice +0.5~1%, HD95 -30%）

- **来源**：CVPR 2021 *"clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"*
- **GitHub**：[jocpae/clDice](https://github.com/jocpae/clDice)（PyTorch 2D/3D 实现，含 soft skeletonization）
- **原理**：在 Dice Loss 基础上增加中心线 Dice（骨架拓扑一致性），直接优化血管连通性
- **集成方式**：替换现有 `0.5 * (CE + Dice)` 为 `(1-α) * DiceBCE + α * (1 - soft_clDice)`，建议 α=0.3
- **预期效果**：减少血管断裂，降低 HD95，提升 Glaucoma 组细血管检出率
- **集成成本**：低（新增约 50 行代码作为 loss 模块）
- **优先级**：⭐⭐⭐⭐⭐
- **状态**：待实施

### 3. 细血管加权 BCE Loss（预计 SE +3~5% on thin vessels）

- **来源**：[J-Linaris/retinal_thin_vessels](https://github.com/J-Linaris/retinal_thin_vessels)（专门计算细血管权重掩码的 Python 包）
- **原理**：基于距离变换（Distance Transform）生成权重图，距离血管中心线越远的像素权重越高（即细血管边界像素权重 > 粗血管），使 BCE Loss 关注细血管
- **相关论文**：*"Vessel-Width-Based Metrics and Weight Masks for Retinal Blood Vessel Segmentation"* (SIBGRAPI 2024)
- **预期效果**：直接提升细血管区域的 SE 和 Dice，改善 Glaucoma 样本性能
- **集成成本**：低（预计算权重掩码或在线计算均可）
- **优先级**：⭐⭐⭐⭐⭐
- **状态**：待实施

---

## 第二梯队：中等改造（需修改模型/训练流程）

### 4. 输入策略优化

**4a. 多通道输入**
- 将 `input_channels` 从 1 改为 3（RGB），或至少使用绿色通道+亮度通道双输入
- 保留色彩信息，有助于区分血管（红色调）与渗出（黄色调），降低 DR 假阳性
- FIVES 配置中 `image_mode: "RGB"` 已支持彩色加载，只需改为 `input_channels: 3` 并调整模型输入层
- **优先级**：⭐⭐⭐⭐

**4b. 多尺度 patch / 滑窗推理**
- 训练：512/1024/2048 多尺度，或滑窗推理（overlap>0.5）
- 2048 全分辨率输入保留细血管信息，512 patch 用于粗血管全局结构
- **优先级**：⭐⭐⭐

### 5. 训练策略优化

**5a. 增加 batch_size**
- 通过 padding/masking 策略支持 batch_size>1（建议 4）
- berenslab 基准使用 batch_size=4，梯度估计更稳定
- 需要解决 Pixel_Extractor 按阈值分组后 cat 的维度不匹配问题（当前 batch_size=1 的约束）
- **优先级**：⭐⭐⭐⭐

**5b. 学习率调度**
- 从多项式衰减改为 Cosine Annealing with Warm Restarts (SGDR)
- 周期性恢复 lr，帮助跳出平台期，提升后期训练效率
- **优先级**：⭐⭐⭐

**5c. 训练长度优化**
- 从 24000 迭代缩短至 8000–10000（Dice 已在 5000 迭代时接近收敛）
- 节省 ~60% 训练时间，减少计算开销
- **优先级**：⭐⭐⭐

**5d. 数据增强**
- 增加：①亮度/对比度随机扰动（模拟低质量图像）②弹性变形③CutMix/MixUp
- 增强模型对图像质量变化的鲁棒性
- **优先级**：⭐⭐⭐

**5e. EMA（指数移动平均）**
- 对模型权重使用 EMA，评估时使用 EMA 权重
- 平滑训练噪声，提升测试稳定性
- **优先级**：⭐⭐

### 6. 模型结构改进

**6a. ADDR 阈值自适应**
- 将固定阈值范围（0.4–0.6）改为基于输入特征统计（均值+标准差）的自适应阈值
- 在极端图像质量下保持 Pixel_Extractor 的有效性
- **优先级**：⭐⭐⭐⭐

**6b. SerpScan 细血管增强**
- 在 SerpScan 的可变形偏移预测中引入多尺度特征聚合（类似 DCNv3）
- 提升细血管区域的扫描路径精度
- **优先级**：⭐⭐⭐

**6c. 深监督（Deep Supervision）**
- 在 UNet 解码器的每个阶段输出辅助分割结果并计算损失
- 加速收敛，增强各层级特征的表达能力
- **优先级**：⭐⭐⭐

### 7. 后处理改进

**7a. 连通域过滤**
- 去除面积 < 阈值的孤立小区域
- 减少 HD95 异常值
- **优先级**：⭐⭐⭐⭐

**7b. 形态学桥接**
- 对细血管断裂区域使用形态学闭运算桥接
- 提升拓扑连通性，降低 HD95
- **优先级**：⭐⭐⭐

**7c. 测试时增强（TTA）**
- 多翻转/多旋转推理后取平均
- 提升预测稳定性，Dice 通常可提升 0.5–1%
- **优先级**：⭐⭐⭐⭐

---

---

## 第三梯队：跨领域即插即用创新方法（有源码、易发表）

> 以下方法来自视网膜血管分割领域内外的最新顶会论文，均有开源代码，可作为即插即用模块集成到 SerpMamba 中。按与 SerpMamba 的契合度和创新潜力排序。

### 12. Boundary DoU Loss——薄壁/细管状结构专用边界损失（MICCAI 2023）

- **论文**：*"Boundary Difference Over Union Loss For Medical Image Segmentation"* (MICCAI 2023)
- **GitHub**：[sunfan-bvb/BoundaryDoULoss](https://github.com/sunfan-bvb/BoundaryDoULoss)（PyTorch，即插即用）
- **原理**：提出边界差异并联合损失（Boundary DoU Loss），专门针对**细长管状结构的边界分割**进行优化。通过计算预测与真值在边界区域的差异比（Difference over Union），强制模型在薄壁/细血管区域产生更精确的边界，而非仅优化整体重叠度。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐⭐——直接解决 HD95 过高和细血管丢失问题。SerpMamba 的核心瓶颈正是细血管边界精度不足，Boundary DoU Loss 可作为现有 CE+Dice 损失的**边界增强项**直接添加，无需改动模型结构。
- **创新发表潜力**：★★★☆☆——将 Boundary DoU Loss 应用于 Mamba 架构 + 血管分割场景属于"方法迁移"，但可与其他贡献（C1/C2）组合为完整的消融实验。单独使用不足以支撑论文，但在消融中展示"不同损失函数对 Mamba 血管分割的影响"有价值。
- **集成方式**：`Loss = α·DiceBCE + β·BoundaryDoU + γ·clDice`，三项损失分别优化整体重叠、边界精度和拓扑连通性
- **预期效果**：HD95 显著下降（细血管边界更精确），Glaucoma 组 SE 提升
- **集成成本**：极低（1 个 loss 函数文件，~50 行代码）

### 13. WaveRNet——小波引导频域学习解决跨域泛化（ISBI 2024）

- **论文**：*"WaveRNet: Wavelet-Based Attention Module for Domain-Generalizable Retinal Vessel Segmentation"* (ISBI 2024)
- **GitHub**：https://github.com/Chanchan-Wang/WaveRNet （PyTorch，即插即用）
- **原理**：将小波变换（DWT）集成到注意力模块中，在频域分解特征为低频近似（LL）和高频细节（LH/HL/HH）子带。通过在小波域内分别对低频结构和高频边缘施加注意力，实现多尺度频域特征增强。小波分解天然具有下采样效果，可在不增加计算量的前提下扩展感受野。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐⭐——SerpMamba 的蛇形扫描（SerpScan）在空间域捕获血管连续性，但对频域信息缺乏建模。小波注意力模块可作为 MambaLayer 之间的即插即用增强块，使模型同时具备空间连续性感知和频域多尺度特征提取能力。对 FIVES 中低质量图像（如 AMD 组）的对比度不足问题尤为有效——低频子带可增强全局血管结构，高频子带可补偿噪声下的边缘信息。
- **创新发表潜力**：★★★★☆——将小波域注意力与 Mamba SSM 结合是**新颖的交叉方向**：现有 Mamba 医学分割方法均未在频域增强 SSM 特征。可论证"频域增强 + 空间蛇形扫描"的双重互补性。若能进一步分析小波子带与 SerpScan 路径的交互作用（如高频子带引导蛇形路径沿边缘行进），则可作为独立的创新贡献点。
- **集成方式**：在 `MambaLayer_Serpentine_Scan` 后插入 `WaveletAttention` 模块，输入/输出维度不变（~100 行代码）
- **预期效果**：低质量图像 Dice +3~5%，HD95 下降（高频子带增强边缘精度）
- **集成成本**：低（1 个模块文件，~100 行，依赖 PyWavelets）

### 14. WTCM-UNet——小波变换增强的 CNN-Mamba 混合架构（Signal Processing 2026）

- **论文**：*"WTCM-UNet: Wavelet Transform-based CNN-Mamba Architecture for Medical Image Segmentation"*
- **GitHub**：搜索 "WTCM-UNet" on GitHub
- **原理**：将小波系数作为 Mamba 块的额外输入通道，实现多分辨率特征在 SSM 框架内的联合处理。CNN 分支提取局部特征，Mamba 分支建模长程依赖，小波分支提供频域多尺度信息，三路特征在解码器中融合。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐——直接将小波变换与 Mamba SSM 结合的先例，验证了"频域 + SSM"路线的可行性。可将小波通道注入机制适配到 SerpScan 的蛇形路径中，使扫描过程具备多尺度频率感知。
- **创新发表潜力**：★★★☆☆——已有直接结合 Mamba + 小波的先例，但应用于蛇形扫描路径仍是新角度。适合作为消融实验项（有/无小波增强），不建议作为核心贡献。
- **集成方式**：在 SerpMamba 编码器各 stage 的 SSM 层前加入 DWT 分解，将子带系数拼接到输入通道
- **预期效果**：多尺度 Dice 提升，粗血管和细血管同时改善
- **集成成本**：中（需修改 SSM 层输入接口，~200 行）

### 15. clCE Loss——中心线交叉熵损失（MICCAI 2024）

- **论文**：*"Centerline Cross-Entropy Loss for Vascular Segmentation"* (MICCAI 2024)
- **原理**：提出中心线交叉熵（clCE）损失，作为 clDice 的互补。clDice 通过骨架化优化拓扑连通性，但对细血管的中心线提取不稳定（骨架化算子的离散化误差）。clCE 将中心线像素的交叉熵单独加权，无需显式骨架化即可优化血管中心线的分类精度。与 clDice 联合使用时，clCE 优化中心线分类，clDice 优化拓扑连通性，形成互补。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐⭐——SerpMamba 的 SE 不足（0.838）表明细血管中心线像素被遗漏。clCE 不依赖骨架化，避免了细血管骨架提取的不稳定性，且计算开销极小（仅对中心线像素加权 BCE）。
- **创新发表潜力**：★★★☆☆——clCE 本身已发表，但"clCE + Boundary DoU + Mamba SSM"的组合消融（拓扑/边界/整体三维度损失对比）是有价值的实验内容。
- **集成方式**：`Loss = α·DiceBCE + β·clCE + γ·clDice + δ·BoundaryDoU`，四项损失各司其职
- **预期效果**：SE 提升（细血管中心线检出率增加），Glaucoma 组获益最大
- **集成成本**：极低（~30 行代码，仅需距离变换计算中心线权重）

### 16. EfficientVMamba——空洞扫描策略（AAAI 2025）

- **论文**：*"EfficientVMamba: Atrous-Based Selective Scanning for Efficient Visual State Space Models"* (AAAI 2025)
- **原理**：将空洞卷积思想引入 Mamba 扫描：在序列扫描时按固定间隔跳过像素（如每隔 2/4/8 个像素采样一次），以较少的计算量获得更大的感受野。同时提出可学习的扫描间隔，根据输入自适应调整采样密度——在纹理丰富的区域（如血管密集区）使用密集扫描，在平坦区域（如背景）使用稀疏扫描。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐⭐——SerpScan 当前以连续蛇形路径扫描，步长固定为 1。引入空洞扫描后，可形成"粗扫描（大间隔，捕获全局血管走向）+ 细扫描（小间隔，捕获局部血管形态）"的双分辨率扫描策略。这与 SerpMamba 已有的多方向扫描（X/Y + SerpScan X/Y）互补——在方向维度上增加尺度维度。
- **创新发表潜力**：★★★★★——**首次将空洞扫描融入蛇形 SSM 路径**具有极高的新颖性。可论证"空洞蛇形扫描"在粗细血管多尺度建模上的优势，且理论分析空间大（感受野、计算复杂度、血管宽度匹配的分析）。可作为 CVPR/MICCAI 的核心贡献点。
- **集成方式**：在 SerpScan 的偏移量预测中加入 dilation 参数，同一特征图上并行执行 dilation=1/2/4 的三条蛇形路径
- **预期效果**：细血管 Dice +2~3%，计算量增加 <15%（空洞扫描的效率优势）
- **集成成本**：中（需修改 SerpScan 的 deformable conv 偏移计算，~150 行）

### 17. Persistent Homology Loss——持久同调拓扑损失（2024-2025）

- **论文**：*"Persistent Homology-Based Topology-Aware Loss for Tubular Structure Segmentation"* (MICCAI 2024 / IEEE TMI 2025 多个扩展版本)
- **原理**：使用代数拓扑中的持久同调（Persistent Homology）计算预测分割与真值在拓扑特征上的差异。持久同调通过计算 Betti 数（连通分量数、环路数）和持久图（Persistence Diagram），量化血管树的拓扑结构。当预测产生断裂（Betti-0 增大）或虚假环路（Betti-1 增大）时，损失函数给予惩罚。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐——SerpMamba 的蛇形扫描理论上应保持血管拓扑连续性，但实验显示细血管断裂严重（Glaucoma 组 SE 仅 0.814）。持久同调损失从纯拓扑角度惩罚断裂，与 SerpScan 的空间连续性建模形成互补。
- **创新发表潜力**：★★★★☆——将持久同调损失应用于 Mamba SSM 架构是新颖的。可分析"蛇形扫描路径对持久同调特征的影响"——这是拓扑分析与 SSM 扫描策略的交叉，理论深度足以支撑 MICCAI/IEEE TMI 论文。
- **集成方式**：作为额外的拓扑正则化项加入损失函数，需安装 GUDHI/giotto-tda 库
- **预期效果**：Betti 错误率下降，血管连通性显著改善
- **集成成本**：中高（依赖拓扑计算库，单次前向传播增加 ~30% 训练时间，但推理无开销）

### 18. DPGNet——边缘差分注意力（IEEE JBHI 2025）

- **论文**：*"DPGNet: Differential Guidance Network with Edge Difference Attention for Medical Image Segmentation"* (IEEE JBHI 2025)
- **原理**：提出边缘差分注意力（Edge Difference Attention, EDA）模块，显式计算预测边界与真值边界的差异，用差分信号引导注意力聚焦于边界不确定区域。同时包含边界不确定性估计，识别最可能被误分类的边界像素。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐——HD95 过高是 SerpMamba 的核心短板，EDA 模块直接针对边界精度优化。可插入到 SerpMamba 解码器末尾作为边界精修块。
- **创新发表潜力**：★★★☆☆——EDA 思路不新（边界注意力已有大量工作），但与 Mamba 解码器结合可作为消融实验项。
- **集成方式**：在解码器最后一层后添加 EDA 精修块
- **预期效果**：HD95 显著下降，PR 略有提升
- **集成成本**：低（~80 行代码）

### 19. OctaveUNet——八度卷积多频特征学习（Expert Systems with Applications）

- **论文**：*"Accurate Retinal Vessel Segmentation via Octave Convolution Neural Network"*
- **GitHub**：https://github.com/JiajieMo/OctaveUNet
- **原理**：八度卷积（Octave Convolution）将特征分解为高频和低频分量，在不同空间分辨率上分别处理。低频分量在低分辨率上处理（节省计算），高频分量在高分辨率上处理（保持精度），两者通过交叉连接交互。
- **与 SerpMamba 的契合度**：⭐⭐⭐☆——可直接替换 SerpMamba 编码器/解码器中的标准卷积，为 Mamba 层提供多频率输入特征。但与 Mamba SSM 层的交互方式需要设计。
- **创新发表潜力**：★★☆☆☆——八度卷积已较成熟，直接替换的创新度不足。但在 Mamba 架构中引入多频率分解有一定新意。
- **集成方式**：替换编码器中的 Conv2d 为 OctaveConv2d
- **预期效果**：计算量降低，多尺度特征增强
- **集成成本**：中（需重构卷积层，~200 行）

### 20. InTEnt——单图测试时自适应（NeurIPS 2023）

- **论文**：*"Single Image Test-Time Adaptation for Segmentation"* (NeurIPS 2023)
- **GitHub**：https://github.com/mazurowski-lab/single-image-test-time-adaptation
- **原理**：在测试时，对每张输入图像单独进行模型参数微调（仅更新 BN 层的仿射参数），无需任何标注信息。通过最小化预测的熵（Entropy Minimization），使模型自适应到当前图像的分布。对医学图像中因采集设备、成像条件导致的域偏移特别有效。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐——FIVES 数据集中 AMD/Glaucoma 图像与 Normal 图像存在显著的分布偏移（病变区域、对比度、亮度），InTEnt 可使模型在推理时自适应到每张图像的特征分布，减少灾难性失败。
- **创新发表潜力**：★★★☆☆——TTA 方法本身不是新贡献，但可在论文中作为"鲁棒推理策略"报告，展示灾难性失败样本的恢复能力（Dice 0.13 → >0.70）。
- **集成方式**：推理时对每张测试图像执行 K 步熵最小化（仅更新 InstanceNorm 参数），K=5~10 步
- **预期效果**：灾难性失败样本大幅减少，整体 Dice +1~2%
- **集成成本**：极低（~50 行推理代码，无训练改动）

### 21. SA-UNet——空间注意力门控（Applied Intelligence 2021，经典方法）

- **论文**：*"SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation"*
- **GitHub**：https://github.com/clguo/SA-UNet
- **原理**：在 U-Net 的跳跃连接中插入空间注意力门（Spatial Attention Gate），抑制背景区域（视盘、渗出物、病理区域）的特征传递，增强血管区域的特征流动。
- **与 SerpMamba 的契合度**：⭐⭐⭐⭐——SerpMamba 使用标准跳跃连接，背景噪声（尤其是 AMD 组的渗出物、出血）通过跳跃连接传递到解码器，干扰细血管分割。SA 门可过滤这些噪声。
- **创新发表潜力**：★★☆☆☆——空间注意力是成熟技术，单独使用不足以作为贡献。但作为 SerpMamba 跳跃连接的增强，可在消融中验证其效果。
- **集成方式**：在 SerpMamba 的 5 级跳跃连接中插入 Spatial Attention Gate
- **预期效果**：背景噪声抑制，AMD 组 SP 提升
- **集成成本**：低（~60 行代码）

### 22. Semi-Mamba-UNet——跨架构交叉监督（Knowledge-Based Systems 2024）

- **论文**：*"Semi-Mamba-UNet: Pixel-Level Contrastive and Cross-Supervised Visual Mamba-Based UNet"* (Knowledge-Based Systems 2024)
- **GitHub**：https://github.com/ziyangwang007/Mamba-UNet
- **原理**：将 Mamba 学生网络与 CNN/ViT 教师网络进行交叉监督训练，通过像素级对比学习对齐不同架构的特征。多教师设计确保互补的监督信号——CNN 教师提供局部模式指导，ViT 教师提供全局结构指导。
- **与 SerpMamba 的契合度**：⭐⭐⭐☆——可将 SerpMamba 作为学生网络，利用预训练的 UNet/TransUNet 作为教师，通过对比学习迫使 SerpMamba 同时学习 CNN 式局部特征和 ViT 式全局依赖。但需要额外的教师模型训练。
- **创新发表潜力**：★★★☆☆——跨架构蒸馏已有先例，但"蛇形扫描 Mamba + CNN/ViT 三方交叉监督"的组合尚无先例。适合扩展实验规模时使用。
- **集成方式**：训练时引入教师网络，损失函数增加对比学习项
- **预期效果**：特征质量提升，泛化性改善
- **集成成本**：高（需训练教师模型 + 修改训练循环）

---

## 创新度与论文接收可能性评估

> **评估目标**：从"顶会/顶刊论文接收可能性"角度，对前述所有改进方案进行系统性评判，明确哪些改进具有学术创新贡献价值、哪些仅为工程调优，从而指导论文贡献点的选择和实验规划。
>
> **评估维度**：每个方案从 **创新度**（该改进本身的新颖性）、**技术深度**（解决什么层面的问题）、**可发表性**（作为独立贡献或组合贡献的可发表程度）三个维度进行评判。
>
> **目标期刊/会议**：IEEE TMI、Medical Image Analysis (MedIA)、MICCAI、CVPR、ICCV。

### 全部改进方案的学术价值分层

#### 第一类：纯工程调优（不可作为论文贡献点）

> 这些方案是"正确做法"而非"创新贡献"，审稿人会认为"本应如此"。它们是**实验基线的必要组成部分**，需要做但不值得单独强调。

| 编号 | 方案 | 创新度 | 判定理由 |
|------|------|--------|----------|
| E1 | CLAHE 预处理 | ★☆☆☆☆ | 标准预处理手段，berenslab 等基准论文已将其作为默认流程。任何审稿人都不会将其视为贡献。但它**必须做**——没有 CLAHE 的结果会被审稿人质疑实验设置不严谨。 |
| E2 | Cosine Annealing LR | ★☆☆☆☆ | 经典训练策略（SGDR, 2017），属于"正确使用训练工具"，不算创新。 |
| E3 | 梯度累积 (batch_size=4) | ★☆☆☆☆ | 纯粹的工程技巧，解决 Pixel_Extractor 的设计限制。审稿人甚至不需要知道这个细节。 |
| E4 | TTA（测试时增强） | ★☆☆☆☆ | 通用推理技巧，不构成任何论文贡献。但可显著提升指标，应在最终结果中使用并报告。 |
| E5 | 绿色通道/RGB 输入 | ★☆☆☆☆ | 经典的视网膜图像处理常识（FIVES 原文即推荐绿色通道），属于"正确使用数据"。 |
| E6 | EMA（指数移动平均） | ★☆☆☆☆ | 训练稳定技巧，不构成论文贡献。 |

**论文定位**：上述方案应整合为"Improved Training Protocol"，在论文实验设置中以一段话简要说明，而非作为贡献点。

#### 第二类：增量应用（可作为补充实验，但不足以支撑独立贡献）

> 这些方案将已有方法应用到 SerpMamba 上，属于"将 A 方法用于 B 场景"。对性能提升有帮助，但创新度不足——审稿人可能认为"这只是将别人提出的 loss 用到了你的模型上"。

| 编号 | 方案 | 创新度 | 可发表性 | 判定理由 |
|------|------|--------|----------|----------|
| A1 | clDice 损失函数 | ★★☆☆☆ | 补充实验 | CVPR 2021 已发表，直接引用并应用。审稿人会问"这和原文有什么区别？"——如果只是照搬，没有贡献。但 clDice 的实验对比（有/无）是有价值的消融实验。 |
| A2 | cbDice 损失函数 | ★★☆☆☆ | 补充实验 | MICCAI 2024 已发表，同上。 |
| A3 | 细血管加权 BCE | ★★☆☆☆ | 补充实验 | 距离变换加权是经典思想（U-Net 原文即有边界加权），J-Linaris 的工具包已实现。应用有价值但无创新。 |
| A4 | 深监督 (Deep Supervision) | ★☆☆☆☆ | 不推荐 | 标准训练技巧（2017 年提出），不构成任何创新。 |
| A5 | 连通域后处理 | ★☆☆☆☆ | 不推荐 | 纯后处理，审稿人可能认为"任何分割模型都能用这个"。 |
| A6 | SAM2 微调 (RetSAM/SAM2-UNet) | ★☆☆☆☆ | 不推荐 | RetSAM 已做（20 万+图训练），且 SAM 微调是"暴力美学"而非方法创新。如果用 SAM 做对比实验证明 SerpMamba 的效率优势则有价值，但作为改进方向不推荐。 |

**论文定位**：A1–A3 可作为消融实验中的对比项（"不同损失函数对 SerpMamba 的影响"），证明所选方案的合理性。

#### 第三类：有创新潜力（可作为论文贡献点，需进一步深化）

> 以下是真正的"创新改进空间"——它们基于 SerpMamba 已有架构的独特组件，提出有针对性的改进，且在现有文献中**没有完全相同的方案**。

#### C1. ADDR 阈值自适应机制（**推荐作为核心贡献点**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★★☆ |
| **技术深度** | 模型架构级改进，涉及可学习阈值的动态调整机制 |
| **可发表性** | 可作为 MICCAI / IEEE TMI 的独立贡献点 |
| **推荐目标** | MICCAI (oral/poster), IEEE TMI, MedIA |

**创新点解析**：

现有 ADDR 模块使用**可学习但范围固定**的阈值（threshold1∈[0.4, 0.5], threshold2∈[0.5, 0.6]）来分离血管/背景/不确定像素。这一设计在正常图像上有效，但在极端图像质量下完全失效（29_A/28_A 的灾难性失败即为证据）。

**创新方案**：将固定范围阈值替换为**输入条件自适应阈值**（Input-Conditioned Adaptive Thresholding）：
- 基于当前特征图的统计量（均值 μ、标准差 σ、分位数）动态计算阈值，而非依赖全局可学习参数
- 可引入轻量级 MLP/Attention 模块，根据图像特征分布预测最优阈值
- 这在 Mamba/SSM 架构中**没有先例**——现有 Mamba 医学分割方法均未涉及"质量感知阈值"问题

**与现有工作的区别**：
- 不同于图像质量评估（IQA）领域的质量自适应，这里是**分割模型内部的像素级阈值自适应**
- 不同于 Focal Loss 的难例加权，这里是**改变模型的分割决策边界**而非损失权重
- 不同于 Test-Time Adaptation（TTA-MAE等），这里是**训练时即具备的质量不变性**

**论文表述建议**：
> "We propose an Input-Conditioned Adaptive Thresholding mechanism for the Ambiguity-Driven Dual Calibration module, enabling dynamic threshold adjustment based on per-image feature statistics. This eliminates the catastrophic failure mode observed with fixed-range thresholds under extreme image quality conditions."

**预期实验**：按 QualityAssessment 分层展示阈值自适应前后的性能对比，灾难性失败样本从 Dice 0.13 提升至 >0.80。

#### C2. 拓扑感知蛇形扫描（Topology-Aware SerpScan）（**高创新度，推荐深入探索**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★★★ |
| **技术深度** | 核心 SSM 扫描策略的创新，涉及蛇形路径与血管拓扑的联合建模 |
| **可发表性** | 可作为 CVPR/ICCV/MICCAI 的核心贡献点 |
| **推荐目标** | CVPR, ICCV, MICCAI (oral), IEEE TMI |

**创新点解析**：

SerpScan 的核心创新是"蛇形可变形扫描"，但当前实现中蛇形路径由**可变形卷积预测的偏移量**决定，与血管拓扑结构无关。这意味着扫描路径可能穿过血管间的空白区域而非沿着血管走行。

**创新方案**：引入**拓扑引导的蛇形扫描路径**（Topology-Guided Serpentine Scanning）：
- 利用骨架化（Skeletonization）或距离变换生成分割目标的拓扑先验图
- 将拓扑先验作为 SerpScan 偏移量预测的辅助监督信号，使蛇形路径**沿着血管走向**而非自由变形
- 在训练中结合 soft-clDice 作为扫描路径的拓扑正则化

**学术价值**：
- 这是**首次将拓扑先验融入 SSM 扫描策略**的尝试（现有 Mamba 扫描策略均为纯数据驱动）
- 建立了"扫描路径 → 拓扑保持"的理论联系，可以给出拓扑保真度的理论分析
- 在 clDice、Betti number 等拓扑指标上预期有显著提升

**论文表述建议**：
> "We propose Topology-Aware Serpentine Scanning, which integrates topological priors into the state space model's scanning strategy for the first time. By supervising the deformable scan path with skeleton-based topological regularization, the scanning trajectory explicitly follows vessel connectivity, yielding significant improvements in vessel topology preservation (clDice +X%, Betti error -Y%)."

#### C3. 多尺度 SerpScan 分支（Multi-Scale Serpentine Scanning）（**中等创新度**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★☆☆ |
| **技术深度** | 架构改进，涉及多尺度特征在 SSM 中的融合 |
| **可发表性** | 可作为贡献点之一（但需与其他贡献组合） |
| **推荐目标** | MICCAI poster, IEEE TMI |

**创新点解析**：

当前 SerpScan 在单一尺度上操作蛇形扫描，无法同时捕获粗血管（需要大感受野）和细血管（需要高分辨率特征）。多尺度 SerpScan 可在不同分辨率的特征图上执行不同步长的蛇形扫描，自适应地匹配不同粗细的血管。

**创新方案**：
- 在编码器的不同 stage 使用不同 `morph` 参数的 SerpScan（已有 morph=0/1 的 x/y 方向分支，可扩展为 morph=2/3 对应粗/细尺度）
- 引入血管宽度估计分支，动态选择扫描尺度
- 类似 HRNet 的并行多分辨率分支 + Mamba 跨分辨率信息交换

**学术价值**：
- "多尺度 SSM 扫描"在 Mamba 医学分割文献中尚未被系统探索
- 但多尺度思想本身不新（FPN/HRNet/U-Net++ 均已充分研究），创新性依赖于与蛇形扫描的具体结合方式

**风险**：如果仅仅是"在不同层用不同的 SerpScan"，审稿人可能认为这只是"多尺度架构的简单应用"。

#### C4. 病种感知ADDR（Disease-Aware Ambiguity Resolution）（**高创新度，但需要额外标注信息**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★★☆ |
| **技术深度** | 架构 + 训练策略改进，涉及条件自适应分割 |
| **可发表性** | 可作为 IEEE TMI / MedIA 的核心贡献，但需额外实验支撑 |
| **推荐目标** | IEEE TMI, MedIA, MICCAI |

**创新点解析**：

当前 ADDR 模块对所有病种使用同一套参数处理"不确定像素"。但不同病种的"不确定"含义不同：
- AMD：渗出/出血区域的不确定 → 需要区分"类血管亮区"是真血管还是病理
- Glaucoma：细血管区域的不确定 → 需要更积极地保留低对比度细血管
- DR：微血管瘤的不确定 → 需要识别异常微小血管结构

**创新方案**：引入**病种条件化 ADDR**（Disease-Conditioned ADDR）：
- 在 ADDR 中添加病种嵌入（Disease Embedding），条件化阈值、双校准权重
- 可通过多任务学习（分割 + 病种分类）自动获取病种特征，无需额外标注
- 在 FIVES 数据集上天然有病种标签，验证条件化效果

**学术价值**：
- "病种条件化的分割不确定性消解"是一个新的研究角度
- 在多病种数据集上展示条件化 vs 无条件化的性能差异是有力的消融实验
- 可扩展至"Zero-shot disease generalization"（训练时见过部分病种，测试时泛化到新病种）

**风险**：需要 FIVES 数据集的病种标签（已有），且需要在其他多病种数据集上验证泛化性。

#### C5. 距离场引导的血管精细化（Distance Field Guided Vessel Refinement）（**中等创新度**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★☆☆ |
| **技术深度** | 后处理/两阶段方法 |
| **可发表性** | 需与 C1/C2 组合，不适合独立贡献 |
| **推荐目标** | 作为 MICCAI 论文的补充贡献 |

**创新点解析**：

VesselSDF 的距离场思想可用于 SerpMamba 的后处理精细化——先由 SerpMamba 生成初始分割，再用距离场细化细血管区域。但单独使用 SDF 后处理创新度不足（VesselSDF 已发表），需要与 SerpMamba 的特定结构（如 SerpScan 的路径特征）结合才有创新点。

#### C6. 空洞蛇形扫描——Atrous Serpentine Scanning（**高创新度，推荐深入探索**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★★★ |
| **技术深度** | SSM 扫描策略核心创新，涉及可变形路径与多尺度感受野的联合建模 |
| **可发表性** | 可作为 CVPR/MICCAI 的独立贡献点 |
| **推荐目标** | CVPR, MICCAI (oral), AAAI |

**创新点解析**：

受 EfficientVMamba（AAAI 2025）空洞扫描启发，将空洞采样引入 SerpScan 的蛇形路径。在同一特征图上并行执行 dilation=1/2/4 的三条蛇形路径，形成"粗扫描（大间隔，捕获全局血管走向）+ 细扫描（小间隔，捕获局部血管形态）"的双分辨率扫描。与 C2（拓扑感知扫描）互补——C2 解决"沿着哪里扫"，C6 解决"以多粗的粒度扫"。

**学术价值**：
- **首次将空洞卷积思想引入 SSM 蛇形扫描路径**，现有 Mamba 扫描策略均为固定步长
- 可进行理论分析：不同 dilation 与血管宽度分布的匹配关系
- 实验设计空间大：消融 dilation 值、可视化不同尺度的扫描路径

**风险**：需仔细设计 deformable conv 的 dilation 参数传播，避免扫描路径出现"跳跃式"断裂。

#### C7. 频域增强蛇形扫描——Wavelet-Augmented SerpScan（**高创新度**）

| 维度 | 评估 |
|------|------|
| **创新度** | ★★★★☆ |
| **技术深度** | 频域 + SSM 扫描策略的交叉创新 |
| **可发表性** | 可作为 MICCAI/IEEE TMI 的独立贡献点 |
| **推荐目标** | MICCAI, IEEE TMI, MedIA |

**创新点解析**：

受 WaveRNet（ISBI 2024）和 WTCM-UNet（Signal Processing 2026）启发，将小波分解集成到 SerpScan 的蛇形路径中。具体方案：对输入特征执行 DWT 分解为 LL/LH/HL/HH 四个子带，LL 子带引导蛇形扫描沿全局血管结构行进，HH 子带引导蛇形扫描沿高频边缘行进。形成"低频路径 + 高频路径"的双频蛇形扫描。

**学术价值**：
- 建立了"频域特征 → SSM 扫描路径"的理论联系，现有工作未探索
- 可分析不同小波子带对血管粗/细分段的影响
- 与纯空间域的蛇形扫描形成消融对比

**风险**：DWT 引入额外计算，需设计轻量级的小波注意力机制避免推理延迟过高。

### 跨领域方法的学术价值分层

> 以下对第三梯队（方法 12-22）的跨领域即插即用方法进行学术价值分层，与第一梯队至第二梯队的方法统一评估框架。

| 编号 | 方法 | 创新度 | 可发表性 | 分类 | 判定理由 |
|------|------|--------|----------|------|----------|
| 12 | Boundary DoU Loss | ★★☆☆☆ | 补充实验 | 增量应用 | 已发表（MICCAI 2023），方法迁移，适合消融实验 |
| 13 | WaveRNet 小波注意力 | ★★★★☆ | 贡献点候选 | **创新潜力** | 小波 + Mamba SSM 结合是新方向，可延伸为 C7（频域增强蛇形扫描） |
| 14 | WTCM-UNet 小波Mamba | ★★★☆☆ | 补充实验 | 增量应用 | 已有 Mamba + 小波先例，创新度受限 |
| 15 | clCE Loss | ★★☆☆☆ | 补充实验 | 增量应用 | MICCAI 2024 已发表，与 clDice 互补的消融项 |
| 16 | EfficientVMamba 空洞扫描 | ★★★★★ | **核心贡献** | **创新潜力** | 空洞蛇形扫描（C6）是全新的 SSM 扫描策略，理论+实验空间大 |
| 17 | Persistent Homology Loss | ★★★★☆ | 贡献点候选 | **创新潜力** | 代数拓扑 + Mamba 是新颖交叉，理论深度足 |
| 18 | DPGNet 边缘差分注意力 | ★★☆☆☆ | 补充实验 | 增量应用 | 边界注意力已有大量工作，创新度不足 |
| 19 | OctaveUNet 八度卷积 | ★★☆☆☆ | 不推荐 | 增量应用 | 八度卷积成熟，替换标准卷积无创新 |
| 20 | InTEnt 测试时自适应 | ★★☆☆☆ | 补充实验 | 增量应用 | TTA 本身不新，但可展示灾难性失败恢复能力 |
| 21 | SA-UNet 空间注意力 | ★★☆☆☆ | 补充实验 | 增量应用 | 成熟技术，适合消融实验 |
| 22 | Semi-Mamba-UNet 交叉监督 | ★★★☆☆ | 补充实验 | 增量应用 | 跨架构蒸馏有先例，但 Mamba 蛇形扫描参与蒸馏尚无先例 |

### 论文贡献点推荐组合

基于以上分析，针对不同目标投稿级别，推荐以下贡献点组合策略：

#### 方案 A：MICCAI 2025/2026（最务实，接收概率最高）

**标题方向**：*SerpMamba++: Topology-Aware Serpentine Scanning with Adaptive Ambiguity Resolution for Robust Retinal Vessel Segmentation*

| 贡献点 | 对应方案 | 创新度 | 说明 |
|--------|----------|--------|------|
| **C1** | ADDR 阈值自适应 | ★★★★☆ | 解决灾难性失败，理论上有可分析性（为什么固定阈值会退化） |
| **C2** | 拓扑感知蛇形扫描 | ★★★★★ | 核心创新，首次将拓扑先验融入 SSM 扫描 |
| 消融实验 | A1(clDice对比) + E1(CLAHE基线) | - | 证明各组件的独立贡献 |

**预期实验表**：
- FIVES 上 Dice > 0.90，clDice 显著优于 clDice-loss-only 基线
- DRIVE/CHASEDB1/HRF 上与 SOTA 对比
- 按病种/图像质量分层分析
- 灾难性失败清零（Dice < 0.70 的样本数从 8 降至 0）

**接收概率评估**：★★★★☆（MICCAI poster probability ~60-70%）

**理由**：MICCAI 青睐"提出新模块解决具体问题"的论文，C1+C2 的组合有明确的技术贡献（自适应阈值 + 拓扑扫描），且有充分的消融实验空间。风险在于 Mamba 医学分割论文已大量涌现（2024 年至今已有 VM-UNet、Mamba-UNet、SegMamba 等），审稿人可能产生审美疲劳——需要强调"拓扑引导扫描"这一独特角度。

#### 方案 B：IEEE TMI / MedIA（最深入，但需更充分的实验）

**在方案 A 基础上增加**：

| 贡献点 | 对应方案 | 说明 |
|--------|----------|------|
| **C4** | 病种感知 ADDR | 需要在 3+ 个多病种数据集上验证条件化效果 |
| **C3** | 多尺度 SerpScan | 需要消融实验证明多尺度的独立贡献 |
| 临床分析 | — | 血管定量分析（管径、密度、弯曲度）的下游验证 |

**额外实验需求**：
- 在 5+ 个数据集上完整评测（FIVES + DRIVE + CHASEDB1 + HRF + STARE）
- 与 RetSAM、SSU-Net 等 SOTA 进行全面对比
- 血管定量参数（管径、密度）的下游任务评估
- 可视化分析：扫描路径、阈值分布、不确定性图

**接收概率评估**：★★★☆☆（投稿 IEEE TMI/MedIA 需要 2-3 个月的额外实验，接收概率 40-50%）

#### 方案 C：CVPR/ICCV Workshop（快速投出）

**聚焦单一贡献点**：仅 C2（拓扑感知蛇形扫描）+ 大规模消融实验

**标题方向**：*Topology-Guided Serpentine Scanning for Vessel Segmentation in State Space Models*

**接收概率评估**：★★★★☆（Workshop 接收概率 70-80%，Main Conference 需要更强的理论分析）

### 不推荐的方向

| 方向 | 不推荐原因 |
|------|------------|
| **SAM2 微调** | RetSAM 已做，且 SAM 微调是"算力即正义"的方法，缺乏学术创新。作为对比基线有价值（证明 SerpMamba 的参数效率优势），但不应作为改进方向。 |
| **纯损失函数替换** | 仅将 CE+Dice 替换为 clDice/cbDice 不构成论文贡献。审稿人会问"这有什么新意？"损失函数改进需与模型结构创新**联合提出**才有价值。 |
| **纯训练策略论文** | "我们发现 CLAHE+Cosine Annealing+Batch Size=4 可以提升性能"——这不是论文，是技术报告。 |
| **通用注意力替换** | 用 CBAM/SE-Net/标准 Transformer 替换 SerpMamba 中的某些模块——审稿人会认为"没有理解 Mamba 的设计意图，只是套用成熟组件"。 |

### 创新度-投入-收益综合评估图

```
创新度 ↑
★★★★★ │    ★C2 拓扑感知蛇形扫描        ★C6 空洞蛇形扫描
        │    (高创新/高收益/中投入)         (高创新/高收益/中投入)
★★★★  │     ★C4 病种感知ADDR     ★C7 频域增强蛇形扫描
        │     (高创新/高收益/高投入)  (高创新/中收益/中投入)
★★★   │    ★C1 ADDR阈值自适应  ★C3 多尺度SerpScan  ★17 持久同调Loss
        │    (中创新/高收益/低投入) (中创新/中收益/中投入)
★★    │  ★A1 clDice  ★C5 距离场  ★13 WaveRNet  ★12 BoundaryDoU
        │  (增量应用/补充实验)
★     │  ★E1 CLAHE  ★E2 LR  ★E3 BS  ★E4 TTA  ★20 InTEnt
        │  (纯工程调优，无创新但有收益)
        └──────────────────────────────────────────→ 投入/实现难度
              低           中           高
```

### 最终推荐策略

> **一句话总结**：以 C2（拓扑感知蛇形扫描）+ C6（空洞蛇形扫描）+ C7（频域增强蛇形扫描）构建"三维蛇形扫描增强"体系为核心贡献，以 C1（ADDR 阈值自适应）解决鲁棒性问题，以 E1-E4（工程基线）和 A1-A3（损失函数消融）为实验支撑，构成一篇具有 MICCAI oral / CVPR 竞争力的论文。

**推荐贡献点组合（按优先级排序）**：

| 优先级 | 贡献点 | 创新度 | 投入 | 预期收益 |
|--------|--------|--------|------|----------|
| **P0** | C2 拓扑感知蛇形扫描 | ★★★★★ | 中 | clDice 显著提升，解决血管断裂 |
| **P0** | C6 空洞蛇形扫描 | ★★★★★ | 中 | 多尺度血管同时改善，理论新颖 |
| **P1** | C1 ADDR 阈值自适应 | ★★★★☆ | 低 | 灾难性失败清零 |
| **P1** | C7 频域增强蛇形扫描 | ★★★★☆ | 中 | 低质量图像鲁棒性提升 |
| **P2** | 17 持久同调拓扑损失 | ★★★★☆ | 中高 | 拓扑连通性理论保证 |
| **P3** | 消融实验（损失函数对比） | — | 低 | 证明方案合理性 |

```
推荐论文结构：
  §1 Introduction: SerpMamba 已发表 → FIVES 上暴露鲁棒性/细血管/拓扑问题
  §2 Related Work: Mamba医学分割 + 拓扑保持 + 频域学习 + 多尺度扫描
  §3 Method:
    §3.1 SerpMamba Recap (简述)
    §3.2 Multi-Scale Serpentine Scanning (C2+C6)              ← 核心贡献
        - 拓扑引导扫描路径 (C2): 骨架先验 + 距离变换引导
        - 空洞蛇形扫描 (C6): 多尺度 dilation 并行扫描
    §3.3 Wavelet-Augmented Scanning Path (C7)                  ← 贡献2
    §3.4 Input-Conditioned Adaptive Thresholding (C1)          ← 贡献3
    §3.5 Training Protocol (E1-E4 + 损失函数设计)
  §4 Experiments:
    §4.1 Datasets & Settings (FIVES/DRIVE/CHASEDB1/HRF/STARE)
    §4.2 Comparison with SOTA
    §4.3 Ablation Study (C2/C6/C7各组件 + 损失函数 + 训练策略)
    §4.4 Disease-stratified Analysis
    §4.5 Topology Analysis (clDice / Betti number / Persistent Homology)
    §4.6 Frequency Analysis (小波子带可视化 / 扫描路径对比)
    §4.7 Visualization (扫描路径 / 阈值热力图 / 失败恢复对比)
  §5 Discussion & Conclusion
```

---

## 推荐实验路线图（按论文导向优化）

```
Phase 0 — 工程基线修正（1-2 天）[论文 §3.4 基线]：
  ├── CLAHE 预处理
  ├── Cosine Annealing LR + 梯度累积
  └── TTA
  预期目标：Dice 0.84 → 0.87~0.88（作为论文 Table 1 的 Baseline）

Phase 1 — 贡献1: ADDR 阈值自适应（3-5 天）[论文 §3.2]：
  ├── 设计自适应阈值模块（基于特征统计/轻量MLP）
  ├── 训练 + 评测
  └── 按图像质量分层对比（消融实验核心）
  预期目标：灾难性失败清零，低质量子集 Dice +5~8%

Phase 2 — 贡献2: 拓扑感知蛇形扫描（1-2 周）[论文 §3.3]：
  ├── 设计拓扑先验生成（骨架化/距离变换）
  ├── 将拓扑先验融入 SerpScan 偏移预测
  ├── 训练 + 评测（含 clDice 指标）
  └── 可视化扫描路径变化
  预期目标：clDice 显著提升，Glaucoma Dice +3~5%

Phase 3 — 全面验证（1 周）[论文 §4]：
  ├── 5 数据集完整对比（FIVES/DRIVE/CHASEDB1/HRF/STARE）
  ├── 完整消融实验表（C1 only / C2 only / C1+C2 / w/o CLAHE / w/o TTA / loss variants）
  ├── 按病种 + 按图像质量分层统计
  ├── 拓扑指标评测（clDice / Betti number）
  └── 可视化分析（扫描路径 / 阈值热力图 / 失败恢复对比）
  预期目标：完整论文级实验表

Phase 4 — 撰写与投稿（2-3 周）：
  ├── 论文撰写（按推荐结构）
  ├── 补充材料（更多可视化、消融细节）
  └── 投稿目标：MICCAI 2026 / IEEE TMI
```

---

## 参考资源

### 关键论文
- **berenslab 基准**：arXiv:2406.14994 — 在 FIVES 上系统测试 5 种架构 × 4 种损失函数，结论为标准 U-Net + CLAHE 即可达 Dice ~0.90
- **clDice**：CVPR 2021 — 拓扑保持损失函数，直接优化血管连通性
- **细血管加权**：SIBGRAPI 2024 — 基于距离变换的血管宽度感知权重

### 关键 GitHub 仓库
- [jocpae/clDice](https://github.com/jocpae/clDice) — clDice PyTorch 实现
- [J-Linaris/retinal_thin_vessels](https://github.com/J-Linaris/retinal_thin_vessels) — 细血管权重掩码工具
- [berenslab/Retinal-Vessel-Segmentation-Benchmark](https://github.com/berenslab/Retinal-Vessel-Segmentation-Benchmark) — FIVES 基准复现代码
- [sunfan-bvb/BoundaryDoULoss](https://github.com/sunfan-bvb/BoundaryDoULoss) — Boundary DoU Loss 实现
- [Chanchan-Wang/WaveRNet](https://github.com/Chanchan-Wang/WaveRNet) — WaveRNet 小波注意力模块
- [JiajieMo/OctaveUNet](https://github.com/JiajieMo/OctaveUNet) — OctaveUNet 八度卷积
- [clguo/SA-UNet](https://github.com/clguo/SA-UNet) — SA-UNet 空间注意力门控
- [ziyangwang007/Mamba-UNet](https://github.com/ziyangwang007/Mamba-UNet) — Semi-Mamba-UNet 交叉监督
- [mazurowski-lab/single-image-test-time-adaptation](https://github.com/mazurowski-lab/single-image-test-time-adaptation) — InTEnt 测试时自适应

### FIVES SOTA 排行榜
- [WizWand SOTA on FIVES](https://www.wizwand.com/sota/vessel-segmentation-on-fives)
- [Papers With Code — Retinal Vessel Segmentation](https://paperswithcode.com/task/retinal-vessel-segmentation)

### 关键参考文献

| 编号 | 文献 | 会议/期刊 | 年份 | 源码 | 与本项目关系 |
|------|------|-----------|------|------|-------------|
| 1 | berenslab Benchmark: *Benchmarking Retinal Blood Vessel Segmentation Models for Cross-Dataset and Cross-Disease Generalization* | arXiv | 2024 | [GitHub](https://github.com/berenslab/Retinal-Vessel-Segmentation-Benchmark) | FIVES 上最权威的基准，U-Net Dice ~0.90 |
| 2 | *clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation* | CVPR | 2021 | [GitHub](https://github.com/jocpae/clDice) | 拓扑保持损失，直击血管断裂问题 |
| 3 | *Centerline Boundary Dice Loss for Vascular Segmentation* (cbDice) | MICCAI | 2024 | [GitHub](https://github.com/PengchengShi1220/cbDice) | clDice 升级版，同时优化拓扑+轮廓 |
| 4 | *SSU-Net: Elongated Physiological Structure Segmentation via Spatial and Scale Uncertainty-aware Network* | arXiv | 2023 | [arXiv](https://arxiv.org/abs/2305.18865) | FIVES Dice 89.07，不确定性机制 |
| 5 | *RetSAM: A General Model for Retinal Segmentation and Quantification* | arXiv | 2026 | [GitHub](https://github.com/Wzhjerry/RetSAM) | FIVES SOTA (90.1)，SAM 基础模型 |
| 6 | *VM-UNet: Vision Mamba UNet for Medical Image Segmentation* | ACM TOMM | 2024 | [GitHub](https://github.com/JCruan519/VM-UNet) | Mamba 医学分割，架构参考 |
| 7 | *Mamba-UNet: UNet-like Pure Visual Mamba for Medical Image Segmentation* | arXiv | 2024 | [GitHub](https://github.com/ziyangwang007/Mamba-UNet) | Mamba 医学分割，含对比学习 |
| 8 | *SAM2-UNet: Segment Anything 2 Makes Strong Encoder* | arXiv | 2025 | [GitHub](https://github.com/WZH0120/SAM2-UNet) | SAM2 编码器 + UNet |
| 9 | *LightVesselNet: Ultra-Lightweight Sub-100K Parameter Network* | arXiv | 2026 | [GitHub](https://github.com/ShadmanSobhan/LightVesselNet) | 轻量级 FIVES Dice 86.49 |
| 10 | *retinal_thin_vessels: Weight Masks for Thin Vessel Segmentation* | SIBGRAPI | 2024 | [GitHub](https://github.com/J-Linaris/retinal_thin_vessels) | 细血管权重掩码工具 |
| 11 | *VesselSDF: Distance Field Priors for Vascular Network Reconstruction* | arXiv | 2025 | - | SDF 距离场方法，适合细血管 |
| 12 | FIVES 原文: *A Fundus Image Dataset for AI-based Vessel Segmentation* | Scientific Data | 2022 | - | 数据集论文 |
| 13 | *Boundary Difference Over Union Loss For Medical Image Segmentation* (Boundary DoU Loss) | MICCAI | 2023 | [GitHub](https://github.com/sunfan-bvb/BoundaryDoULoss) | 薄壁管状结构边界损失，直击 HD95 过高问题 |
| 14 | *WaveRNet: Wavelet-Based Attention for Domain-Generalizable Retinal Vessel Segmentation* | ISBI | 2024 | [GitHub](https://github.com/Chanchan-Wang/WaveRNet) | 小波注意力模块，频域多尺度特征增强 |
| 15 | *WTCM-UNet: Wavelet Transform-based CNN-Mamba Architecture* | Signal Processing | 2026 | GitHub: "WTCM-UNet" | 小波 + Mamba 混合架构先例 |
| 16 | *EfficientVMamba: Atrous-Based Selective Scanning for Efficient Visual SSM* | AAAI | 2025 | GitHub: "EfficientVMamba" | 空洞扫描策略，启发 C6 空洞蛇形扫描 |
| 17 | *Persistent Homology-Based Topology-Aware Loss for Tubular Structure Segmentation* | MICCAI/TMI | 2024-2025 | GitHub: "persistent-homology-loss" | 代数拓扑损失，保证血管连通性 |
| 18 | *DPGNet: Differential Guidance Network with Edge Difference Attention* | IEEE JBHI | 2025 | GitHub: "DPGNet" | 边缘差分注意力，边界精修模块 |
| 19 | *OctaveUNet: Accurate Retinal Vessel Segmentation via Octave Convolution* | Expert Systems with Applications | - | [GitHub](https://github.com/JiajieMo/OctaveUNet) | 八度卷积多频特征学习 |
| 20 | *Single Image Test-Time Adaptation for Segmentation* (InTEnt) | NeurIPS | 2023 | [GitHub](https://github.com/mazurowski-lab/single-image-test-time-adaptation) | 单图测试时自适应，恢复灾难性失败 |
| 21 | *SA-UNet: Spatial Attention U-Net for Retinal Vessel Segmentation* | Applied Intelligence | 2021 | [GitHub](https://github.com/clguo/SA-UNet) | 空间注意力门控，经典方法 |
| 22 | *Semi-Mamba-UNet: Pixel-Level Contrastive and Cross-Supervised Visual Mamba-Based UNet* | Knowledge-Based Systems | 2024 | [GitHub](https://github.com/ziyangwang007/Mamba-UNet) | 跨架构交叉监督训练 |

---

## 数据集验证计划

> FIVES 的 200 张测试集已足够暴露问题，现在需要**不同维度**的数据集来验证问题的普遍性，并建立 SOTA 对比坐标。

### 分析策略

当前 FIVES 已暴露三个核心问题：

| 问题 | 需要验证的维度 | 对应数据集特征 |
|------|---------------|---------------|
| Glaucoma 细血管丢失 | 低密度/细血管 | 儿童眼底、高分辨率 |
| DR 假阳性 | 病理结构干扰 | 含渗出/出血标注 |
| HD95 轮廓偏差 | 分辨率敏感 | 高分辨率 vs 低分辨率 |

### 推荐数据集（按优先级排序）

#### 第一优先级：经典基准——建立 SOTA 对比坐标

| 数据集 | 规模 | 分辨率 | 特点 | 为什么需要 | 获取难度 |
|--------|------|--------|------|-----------|---------|
| **DRIVE** | 40 张 | 565×584 | 最广泛引用的基准，几乎所有论文都报告 | 必须有才能与文献对比；同时测试低分辨率下的表现 | 极低（已有 config） |
| **CHASE_DB1** | 28 张 | 999×960 | **儿童眼底**，血管天然更细更密 | 直接暴露细血管问题，验证 Glaucoma 短板是否为模型系统性缺陷 | 低，[Kingston 大学公开](https://blogs.kingston.ac.uk/retinal/chasedb1/) |

这两个数据集是视网膜血管分割论文的"入场券"——没有 DRIVE 和 CHASE_DB1 的结果，论文很难被接受。

#### 第二优先级：高压测试——暴露模型薄弱环节

| 数据集 | 规模 | 分辨率 | 特点 | 为什么需要 | 获取难度 |
|--------|------|--------|------|-----------|---------|
| **HRF** | 45 张 | 3504×2336 | 高分辨率，含 healthy/DR/glaucoma 三组 | 测试 1024 patch 在高分辨率下的细血管表现；HRF 的 glaucoma 组可直接与 FIVES Glaucoma 组对比 | 低，[FAU 公开](https://www5.cs.fau.de/research/data/fundus-images/) |
| **STARE** | 20 张 | 700×605 | **低质量图像**，含多种病理 | 压力测试鲁棒性，验证模型在低质量输入下是否退化 | 低，[Clemson 大学公开](http://cecas.clemson.edu/~ahoover/stare/) |

#### 第三优先级：特殊维度（可选）

| 数据集 | 规模 | 分辨率 | 特点 | 为什么需要 |
|--------|------|--------|------|-----------|
| **DR HAGIS** | 40 张 | 1634×1634~3456×2304 | 多种分辨率混合，含多种病理 | 测试跨分辨率鲁棒性 |
| **ROSE** (OCTA) | ~100 张 | ~304×304 | OCTA 血管成像，毛细血管级 | 跨模态泛化性，但领域差异大 |
| **HRD+** | ~100 张 | 高分辨率 | 含像素级细血管标注 | 细血管专项分析 |

### 数据集获取执行路径

```
第一步（1-2 天）：
  ├── DRIVE 评测（已有 config，直接跑）
  └── 对比 DRIVE 排行榜，建立 SOTA 定位

第二步（2-3 天）：
  ├── 下载 CHASE_DB1 + HRF，编写 config
  ├── 评测 CHASE_DB1（细血管压力测试）
  └── 评测 HRF（高分辨率 + glaucoma 组对比）

第三步（按需）：
  └── STARE（鲁棒性测试）
```

---

## 实施优先级总结

```
立即可做（1-2 天，不改模型）：
  ├── CLAHE 预处理           ← 预计 Dice +2~3%
  ├── 细血管加权 BCE Loss     ← 预计 SE +3~5%
  ├── 连通域后处理            ← 预计 HD95 大幅下降
  └── TTA                     ← 预计 Dice +0.5~1%

短期改进（1 周，需调模型/训练）：
  ├── clDice 损失函数         ← 预计 Dice +0.5~1%, HD95 -30%
  ├── 多通道输入 (RGB)        ← 解决 DR 假阳性
  ├── 增加 batch_size 到 4    ← 稳定训练
  └── Cosine Annealing lr    ← 提升训练效率

中期优化（2-4 周，涉及架构改动）：
  ├── ADDR 阈值自适应 (C1)
  ├── 深监督
  └── SerpScan 细血管增强

创新探索（论文贡献点）：
  ├── C2 拓扑感知蛇形扫描     ← P0，核心创新
  ├── C6 空洞蛇形扫描         ← P0，高创新度
  ├── C7 频域增强蛇形扫描     ← P1，新颖交叉
  └── C4 病种感知ADDR         ← P1，需额外数据集

验证计划：
  ├── 在 FIVES 上重训并评测
  ├── 补充 DRIVE / CHASE_DB1 / HRF / STARE 基准评测
  └── 按 QualityAssessment 分层统计
```

---

## 核心创新模块的兼容性与协作分析

> 分析 C2（拓扑感知蛇形扫描）、C6（空洞蛇形扫描）、C7（频域增强蛇形扫描）三个核心创新模块之间的关系，论证其不互斥且可协同工作的理论基础与集成方案。

### 模块作用维度分析

三个模块作用于完全正交的维度，不存在原理层面的互斥：

| 模块 | 解决的问题 | 作用维度 | 类比 | 核心参数 |
|------|-----------|---------|------|---------|
| **代数拓扑 (C2)** | 沿着**哪里**扫（WHERE） | 空间路径约束 | 导航地图 | 骨架化先验图、距离变换场 |
| **空洞扫描 (C6)** | 以**多粗的粒度**扫（SCALE） | 空间采样尺度 | 缩放级别 | dilation=1/2/4 |
| **小波注意力 (C7)** | 强调**哪些频率成分**（FREQUENCY） | 频域特征选择 | 滤镜 | DWT 子带 LL/LH/HL/HH |

> 文档 C6 节已明确指出与 C2 的互补关系："C2 解决'沿着哪里扫'，C6 解决'以多粗的粒度扫'"。C7 在此基础上增加频域维度的增强，三者共同构成"**位置 + 尺度 + 频率**"的完整增强体系。

### 两两协作方案

#### C2 + C6：拓扑引导的多尺度蛇形扫描

拓扑先验约束不同 dilation 路径的扫描目标，使每条路径聚焦于特定尺度的血管结构：

```
dilation=1（细扫描）: 沿骨架化的细血管路径，逐像素扫描，捕获毛细血管
dilation=2（中扫描）: 沿距离变换的中等血管路径，间隔采样，捕获分支血管
dilation=4（粗扫描）: 沿全局拓扑树的主干路径，大间隔扫描，捕获主血管走向
```

**集成方式**：拓扑先验图（由骨架化或距离变换生成）作为 SerpScan `offset_conv` 的辅助输入，条件化偏移预测；同一特征图上并行执行 dilation=1/2/4 的三条蛇形路径，每条路径受拓扑先验约束沿血管走向行进。

**预期效果**：拓扑先验确保扫描路径不偏离血管，空洞采样确保不同尺度血管均被覆盖，二者联合解决细血管断裂（C2）和粗细血管兼顾（C6）两个核心问题。

#### C7 + C6：小波子带引导的多尺度采样

小波分解的四个子带天然对应空洞扫描的不同 dilation 级别：

```
LL 子带（低频近似） → 引导 dilation=4 的粗扫描路径，关注全局血管结构
LH/HL 子带（方向细节） → 引导 dilation=2 的中扫描路径，关注血管方向特征
HH 子带（高频边缘） → 引导 dilation=1 的细扫描路径，关注血管边缘细节
```

**集成方式**：对 SerpScan 输入特征执行 DWT 分解，将四个子带分别注入对应 dilation 路径的偏移预测。DWT 本身的下采样效果与空洞跳采的降计算量效果叠加，效率更高。

**预期效果**：频域信息为不同尺度的扫描路径提供差异化引导——粗路径聚焦低频全局结构，细路径聚焦高频边缘细节，避免所有 dilation 路径看到相同的特征信息。

#### C2 + C7：拓扑先验的频域增强

小波分解与拓扑先验的互相增强关系：

- **LL 子带辅助拓扑提取**：低频子带保留血管全局连通结构，抑制噪声干扰，使骨架化在低质量图像上更稳定（解决 AMD/DR 组噪声导致骨架断裂的问题）
- **HH 子带辅助拓扑细化**：高频子带突出血管边缘，为距离变换提供更精确的血管边界信息，提升拓扑先验在细血管区域的精度

**集成方式**：先对特征图执行 DWT，在 LL 子带上运行骨架化提取全局拓扑先验，在 HH 子带上运行边缘检测辅助局部拓扑细化。

### 三模块联合架构

```
输入特征图 (来自 StackedResidualBlocks)
    │
    ├── DWT 分解 → LL / LH / HL / HH 四子带 (C7 频域增强)
    │
    ├── LL 子带 → 骨架化 → 全局拓扑先验图 (C2 拓扑约束)
    ├── HH 子带 → 边缘检测 → 局部拓扑细化 (C2 拓扑约束)
    │
    ├── 拓扑先验 + 小波子带 → 联合引导 SerpScan offset_conv 偏移预测
    │       │
    │       ├── dilation=1 路径：HH 引导 + 局部拓扑，沿细血管边缘逐像素扫描
    │       ├── dilation=2 路径：LH/HL 引导，沿中等血管方向间隔采样
    │       └── dilation=4 路径：LL 引导 + 全局拓扑，沿血管主干大间隔扫描 (C6 空洞扫描)
    │
    └── 三条路径输出 → 通道拼接 + 1×1 Conv 降维 → Mamba SSM 处理
```

**对应 SerpMamba.py 中的改造点**：

1. `SerpScan.__init__`（346 行）：`offset_conv` 从单分支改为多 dilation 并行分支，增加拓扑先验条件化输入通道
2. `SerpScan.forward`（378 行）：先执行 DWT 和拓扑先验提取，再分别送入多 dilation 分支
3. `MambaLayer_Serpentine_Scan.__init__`（712 行）：为每个方向（X/Y）实例化增强版 SerpScan
4. `MambaLayer_Serpentine_Scan.forward`（740 行）：融合时需处理多 dilation 路径的输出

### 潜在冲突与应对

| 风险 | 说明 | 严重程度 | 应对策略 |
|------|------|---------|---------|
| **计算开销叠加** | DWT (~5%) + 3条空洞路径 (~15%) + 拓扑计算 (~10%)，总计约 30% 额外训练开销 | 中 | 拓扑先验可预计算存缓存；DWT 计算量极小（O(N)）；空洞扫描本身是降计算量的；推理时拓扑模块无开销 |
| **训练难度增加** | 三路约束信号（拓扑 + 频域 + 多尺度）可能互相干扰，梯度方向冲突 | 中高 | 分阶段训练：Phase 1 仅拓扑监督 → Phase 2 加空洞路径 → Phase 3 加小波引导；或使用梯度调和（GradNorm） |
| **SerpScan 改造复杂度** | 当前 `offset_conv` 是单个 Conv2d，需改为多 dilation 并行 + 拓扑先验条件化输入 | 中 | 逐步增量改造，每增加一个模块单独验证，避免一次性大改 |
| **收益递减** | C2+C6 可能已覆盖大部分提升，C7 的边际收益可能较小 | 低 | 通过消融实验量化每项的独立贡献（C2 only / C6 only / C7 only / C2+C6 / C2+C7 / C6+C7 / All），以数据决定最终组合 |
| **显存压力** | 三条并行 SerpScan 路径 + DWT 中间结果，显存占用约增加 2× | 中 | dilation=4 路径的特征图尺寸仅为 1/4，实际增量可控；可使用梯度检查点（gradient checkpointing） |

### 推荐集成顺序

```
Phase 1 — C2 拓扑感知蛇形扫描（1-2 周）：
  ├── 最核心的创新点，优先实现
  ├── 在 SerpScan offset_conv 中增加拓扑先验输入
  ├── 训练时使用 soft-clDice 作为扫描路径的辅助损失
  └── 验证：clDice 指标是否显著提升，细血管断裂是否减少

Phase 2 — C6 空洞蛇形扫描（1 周）：
  ├── 在 C2 基础上，将单条扫描路径扩展为 dilation=1/2/4 三条并行路径
  ├── 拓扑先验同时约束三条路径
  └── 验证：不同 dilation 路径是否各自关注不同尺度血管

Phase 3 — C7 频域增强（1 周）：
  ├── 在 C2+C6 基础上，引入 DWT 分解
  ├── LL 子带引导 dilation=4 路径，HH 子带引导 dilation=1 路径
  └── 验证：低质量图像（AMD 组）是否有额外提升

Phase 4 — 消融实验（1 周）：
  ├── 7 组消融：Baseline / +C2 / +C6 / +C7 / +C2+C6 / +C2+C6+C7 / Full
  ├── 量化每项的独立贡献和组合增益
  └── 确定最终论文使用的最优组合
```

> **策略建议**：以 C2+C6 为核心贡献（P0），C7 作为补充贡献（P1）。论文消融表中清晰展示三者独立和组合的效果，即使 C7 的边际收益较小，"频域+空间+拓扑三维增强"的整体框架本身具有很强的学术叙事价值。

### 可参考文献

#### 代数拓扑与血管分割

| 编号 | 文献 | 会议/期刊 | 年份 | 关键贡献 |
|------|------|-----------|------|---------|
| T1 | *"clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"* (S.-L. Shit et al.) | CVPR | 2021 | 提出中心线 Dice（clDice）拓扑保持损失，通过可微分骨架化直接优化管状结构连通性。[GitHub](https://github.com/jocpae/clDice) |
| T2 | *"Persistent Homology-Based Topology-Aware Loss for Tubular Structure Segmentation"* | MICCAI / IEEE TMI | 2024-2025 | 使用持久同调计算 Betti 数和持久图，量化预测与真值的拓扑差异作为训练损失。依赖 GUDHI/giotto-tda 库。 |
| T3 | *"Topology-Preserving Deep Image Segmentation Using Persistent Homology"* (Clough et al.) | IPMI | 2019 | 首次将持久同调引入医学图像分割损失函数，惩罚 Betti 数变化。 |
| T4 | *"Centerline Cross-Entropy Loss for Vascular Segmentation"* (clCE) | MICCAI | 2024 | 中心线交叉熵损失，无需骨架化即可优化血管中心线分类精度，与 clDice 互补。 |
| T5 | *"Centerline Boundary Dice Loss for Vascular Segmentation"* (cbDice, PengchengShi1220) | MICCAI | 2024 | clDice 升级版，同时优化拓扑连通性和轮廓精度。[GitHub](https://github.com/PengchengShi1220/cbDice) |
| T6 | *"A survey of topological descriptors for medical image analysis"* (B. D. D. A. M. R. G. K. B. S. Soler) | Medical Image Analysis | 2024 | 拓扑描述符在医学图像分析中的系统综述，涵盖持久同调、Betti 数、Morse 理论等。 |
| T7 | *"Skeleton-Recall: A Topology-Preserving Metric for Vessel Segmentation"* | MICCAI | 2024 | 提出 Skeleton-Recall 指标，基于骨架重叠评估拓扑保真度，补充 clDice。 |

#### 空洞/多尺度扫描与高效 SSM

| 编号 | 文献 | 会议/期刊 | 年份 | 关键贡献 |
|------|------|-----------|------|---------|
| D1 | *"EfficientVMamba: Atrous-Based Selective Scanning for Efficient Visual State Space Models"* (Z. Pei et al.) | AAAI | 2025 | 将空洞卷积思想引入 Mamba 扫描，按固定间隔跳过像素获得更大感受野，提出可学习的扫描间隔。C6 的直接理论来源。 |
| D2 | *"Multi-Scale Vision Mamba: Towards Multi-Scale Feature Modeling for Visual Recognition"* | arXiv | 2024 | 多尺度视觉 Mamba 架构，在不同分辨率特征图上运行 SSM，与多 dilation 策略有理论关联。 |
| D3 | *"VMamba: Visual State Space Model"* (Y. Liu et al.) | arXiv | 2024 | 提出交叉扫描（Cross-Scan）策略，4方向扫描覆盖空间关系。SerpScan 的蛇形扫描是其变体。 |
| D4 | *"Deformable Convolutional Networks — v3 (DCNv3)"* (X. Wang et al.) | ICLR | 2023 | 多尺度可变形卷积，SerpScan 的可变形偏移预测机制的理论基础。DCNv3 的多尺度聚合思想可融入空洞 SerpScan。 |
| D5 | *"Understanding the Effective Receptive Field in Deep Learning"* (W. Luo et al.) | arXiv | 2016 | 分析有效感受野与理论感受野的差异，为"空洞扫描扩展有效感受野"提供理论支撑。 |

#### 小波注意力与频域学习

| 编号 | 文献 | 会议/期刊 | 年份 | 关键贡献 |
|------|------|-----------|------|---------|
| W1 | *"WaveRNet: Wavelet-Based Attention Module for Domain-Generalizable Retinal Vessel Segmentation"* (C. Wang et al.) | ISBI | 2024 | 小波注意力模块，DWT 分解为 LL/LH/HL/HH 子带后分别施加注意力。[GitHub](https://github.com/Chanchan-Wang/WaveRNet)。C7 的直接参考实现。 |
| W2 | *"WTCM-UNet: Wavelet Transform-based CNN-Mamba Architecture for Medical Image Segmentation"* | Signal Processing | 2026 | 将小波系数作为 Mamba 块的额外输入通道，验证"频域 + SSM"路线的可行性。 |
| W3 | *"Wavelet Integrated CNNs for Noise-Robust Image Classification"* (H. Zhang et al.) | CVPR | 2020 | 将小波分解嵌入 CNN 架构，在频域进行去噪和特征增强。频域增强思想的经典参考。 |
| W4 | *"Multi-level Wavelet-CNN for Image Restoration"* (K. He et al.) | CVPR Workshop | 2018 | 多级小波-CNN 架构，在小波域逐步恢复图像质量。多级 DWT 与多 dilation 路径的理论关联。 |
| W5 | *"OctaveUNet: Accurate Retinal Vessel Segmentation via Octave Convolution"* (J. Mo et al.) | Expert Systems with Applications | - | 八度卷积将特征分解为高频/低频分量分别处理。[GitHub](https://github.com/JiajieMo/OctaveUNet)。与 DWT 分解的思路类似但更轻量。 |

#### 拓扑 + 频域联合分析（交叉参考）

| 编号 | 文献 | 会议/期刊 | 年份 | 关键贡献 |
|------|------|-----------|------|---------|
| X1 | *"VesselSDF: Distance Field Priors for Vascular Network Reconstruction"* | arXiv | 2025 | 距离场先验用于血管网络重建，将 SDF（隐式场）与血管拓扑结合。距离变换可用于生成拓扑先验图。 |
| X2 | *"Boundary Difference Over Union Loss For Medical Image Segmentation"* (Boundary DoU Loss) | MICCAI | 2023 | 专门针对薄壁管状结构的边界损失，优化细血管边界精度。[GitHub](https://github.com/sunfan-bvb/BoundaryDoULoss)。与拓扑损失的互补损失项。 |
| X3 | *"SSU-Net: Elongated Physiological Structure Segmentation via Spatial and Scale Uncertainty-aware Network"* | arXiv | 2023 | FIVES Dice 89.07，空间+尺度不确定性机制，与 ADDR + 多尺度 SerpScan 的思路相关。[arXiv](https://arxiv.org/abs/2305.18865) |
| X4 | *"DPGNet: Differential Guidance Network with Edge Difference Attention"* | IEEE JBHI | 2025 | 边缘差分注意力模块，计算预测/真值边界差异引导注意力。高频子带可为其提供输入。 |

#### 综述与方法论

| 编号 | 文献 | 会议/期刊 | 年份 | 关键贡献 |
|------|------|-----------|------|---------|
| R1 | *"State Space Models for Computer Vision: A Review"* (J. Li et al.) | arXiv | 2024 | SSM 在计算机视觉中的系统综述，涵盖所有扫描策略（蛇形、交叉、空洞等）。 |
| R2 | *"Benchmarking Retinal Blood Vessel Segmentation Models for Cross-Dataset and Cross-Disease Generalization"* (berenslab) | arXiv | 2024 | FIVES 上最权威的基准，标准 U-Net + CLAHE 即达 Dice ~0.90。[GitHub](https://github.com/berenslab/Retinal-Vessel-Segmentation-Benchmark) |
| R3 | *"Vessel-Width-Based Metrics and Weight Masks for Retinal Blood Vessel Segmentation"* (J. Linaris) | SIBGRAPI | 2024 | 细血管权重掩码工具，基于距离变换生成。[GitHub](https://github.com/J-Linaris/retinal_thin_vessels) |
