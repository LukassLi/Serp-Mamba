# SerpMamba 改进方案综合排序

> 基于 FIVES（200 张测试）+ CHASE_DB1（7 张测试）跨数据集实验，结合 `docs/methodPaper/` 5 篇文献及网络搜索，按综合评分排序所有改进方案。
>
> **SerpMamba 核心问题**：①细血管检测（FIVES Glaucoma 灾难性失败 + CHASE_DB1 SE 偏低）②血管断裂/clDice 偏低（CHASE_DB1 0.799 vs SOTA 83.62）③HD95 过高（归一化 ~1% 对角线）④模型保守（PR > SE，漏检多于误检）

---

## 评分体系（维度合并版）

原 8 维度合并为 5 维度，消除概念重叠：
- **问题针对性 + 性能提升 → 问题解决力**：性能是问题解决的量化指标，因果关联
- **创新度 + 发表潜力 → 创新性**：创新决定发表价值，高度相关
- **源码 + 易实现 → 工程可行性**：两者共同决定"能否落地"，实际考量不可分
- **顶会、契合度独立保留**：顶会=来源可信度，契合度=领域相关性，测度不同

**总分 = 3×(创新性+问题解决力)/2 + 2×(顶会+契合度)/2 + 工程可行性 + 均衡性**（满分 35）

| 维度 | 权重 | 分值 | 说明 |
|------|------|------|------|
| 创新性 | ×3 | 1-5 | 方法新颖性 + 能否作为论文贡献点 |
| 问题解决力 | ×3 | 1-5 | 对 SerpMamba 核心问题的针对性 + 预期定量改善 |
| 工程可行性 | ×1 | 1-5 | 源码可用性 + 集成复杂度：5=<1天，4=1-3天，3=3-7天，2=1-2周，1=>2周+CUDA |
| 顶会 | ×2 | 0/3/5 | 5=CVPR/MICCAI/TMI/MedIA/NeurIPS/ICML/AAAI，3=一般期刊，0=原创/无 |
| 契合度 | ×2 | 1-5 | 5=血管/Mamba 专用，3=医学影像通用，1=通用视觉 |
| **均衡性** | **×1** | **0-5** | **5 - std(上述 5 维度分值)**，惩罚偏科方法 |

> **均衡性设计意图**：创新高但无可信来源（或反之）的方法会被拉低。各维度 3-4 分的均衡方法 std 小→均衡性高。SvAttn（各维度 3-5，均衡性 4.60）是均衡方法的典范。

---

## 综合排名总表

| # | 方法 | 总分 | 均衡 | 类型 | 针对问题 | 核心信息 |
|---|------|------|------|------|---------|---------|
| 1 | TopoMask 拓扑掩码 | 29.3 | 4.25 | 训练 | ②血管断裂 | IEEE TMI'25，backbone 无关，双相掩码调度 |
| 2 | GLCP Loss 图连通性 | 29.3 | 4.25 | 损失 | ②血管断裂 | MICCAI'25 Oral，图结构连通性保持 |
| 3 | clDice 损失函数 | 29.3 | 3.80 | 损失 | ②血管断裂 | CVPR'21，[GitHub](https://github.com/jocpae/clDice)，骨架拓扑一致性 |
| 4 | Connectivity Loss | 28.8 | 4.25 | 损失 | ②血管断裂 | IEEE TMI'25，连通域差异惩罚，~30 行 |
| 5 | SvAttn 尺度变异注意力 | 27.6 | **4.60** | 架构 | ①细血管消失 | arXiv'26，[GitHub](https://github.com/anthonyweidai/SvANet)，跨尺度追踪 |
| 6 | Boundary Loss | 27.4 | 3.90 | 损失 | ③HD95过高 | MedIA'21，[GitHub](https://github.com/LIVIAETS/boundary-loss)，距离图边界优化 |
| 7 | TAGC 管状感知门控 | 27.3 | 4.25 | 架构 | ②管状保持 | Multimedia Systems'25，Mamba 前置管状感知 |
| 8 | clCE Loss | 26.9 | 3.90 | 损失 | ①细血管中心线 | MICCAI'24，无需骨架化的中心线 CE |
| 9 | Boundary DoU Loss | 26.9 | 3.90 | 损失 | ③HD95+细边界 | MICCAI'23，[GitHub](https://github.com/sunfan-bvb/BoundaryDoULoss)，薄壁管状边界 |
| 10 | 注意力门控跳跃连接 | 26.9 | 3.90 | 架构 | ①②解码器噪声 | MICCAI'19，[GitHub](https://github.com/ozan-oktay/Attention-Gated-Network)，替换 torch.cat |
| 11 | FcaNet 频域通道注意力 | 26.3 | 4.25 | 架构 | ③频域增强 | ICCV'21，[GitHub](https://github.com/cfzd/FcaNet)，DCT 替代 SE |
| 12 | 细血管加权 BCE | 26.3 | 3.83 | 损失 | ①细血管+④保守 | SIBGRAPI'24，[GitHub](https://github.com/J-Linaris/retinal_thin_vessels)，距离变换加权 |
| 13 | EfficientVMamba 空洞扫描 | 26.3 | 3.83 | 扫描 | ①多尺度 | AAAI'25，空洞 SSM 扫描 → C6 灵感来源 |
| 14 | ADDR 阈值自适应 | 26.0 | 3.04⚠ | 架构 | ①灾难性失败 | **原创**，输入条件化自适应阈值 |
| 15 | WaveRNet 小波注意力 | 25.7 | 4.20 | 架构 | ③低质量鲁棒 | ISBI'24，[GitHub](https://github.com/Chanchan-Wang/WaveRNet)，DWT 子带注意力 |
| 16 | CoordAtt 坐标注意力 | 25.3 | 3.83 | 架构 | ①方向感知 | CVPR'21，[GitHub](https://github.com/Andrew-Qibin/CoordAttention)，x/y方向编码 |
| 17 | InTEnt 测试时自适应 | 25.0 | 3.98 | 推理 | ①灾难性失败 | NeurIPS'23，[GitHub](https://github.com/mazurowski-lab/single-image-test-time-adaptation)，单图 BN 微调 |
| 18 | 拓扑感知蛇形扫描 | 24.9 | 2.94⚠ | 扫描 | ②血管断裂 | **原创**，拓扑先验引导 SerpScan 路径 |
| 19 | AG-TAL 管状感知损失 | 24.6 | 3.14 | 损失 | ①④细血管+断裂 | 半径感知 Dice + 断裂感知 clDice |
| 20 | DCNv4 可变形卷积 | 24.0 | 3.98 | 架构 | 扫描效率 | CVPR'24，[GitHub](https://github.com/OpenGVLab/DCNv4)，3×加速 |
| 21 | Focal Tversky Loss | 24.0 | 3.98 | 损失 | ④保守偏差 | ISBI'19，[GitHub](https://github.com/nabsabraham/focal-tversky-unet)，可调 FN 惩罚 |
| 22 | ASPP 空洞金字塔 | 24.0 | 3.50 | 架构 | ①多尺度血管 | ECCV'18 经典，多速率并行空洞卷积 |
| 23 | 频域增强蛇形扫描 | 23.6 | 3.06⚠ | 扫描 | ③低质量鲁棒 | **原创**，DWT 子带引导 SerpScan |
| 24 | 病种感知 ADDR | 23.3 | 3.28 | 架构 | ①病种差异 | **原创**，病种嵌入条件化阈值 |
| 25 | DFF 方向特征融合 | 22.8 | 4.25 | 架构 | 解码器方向 | Multimedia Systems'25，GAP+Sigmoid 方向权重 |
| 26 | DPGNet 边缘差分注意力 | 22.4 | 4.37 | 架构 | ③边界 | IEEE JBHI'25，边界差异引导注意力 |
| 27 | WTCM-UNet 小波Mamba | 22.4 | 4.37 | 架构 | 频域+SSM | Signal Processing'26，小波系数注入 Mamba |
| 28 | CLAHE 预处理 | 22.2 | 3.67 | 预处理 | 全局基础 | berenslab 验证 Dice +2-3%，1 行代码 |
| 29 | HMS-VesselNet 难例挖掘 | 21.0 | 3.53 | 训练 | ①困难样本 | 难例挖掘 + clDice 联合优化 |
| 30 | Polygon-Mamba | 20.8 | 3.28 | 扫描 | 扫描路径 | PS-VSS 多边形扫描 + SFCAM |
| 31 | SA-UNet 空间注意力 | 20.7 | 3.74 | 架构 | 背景噪声 | Applied Intelligence'21，[GitHub](https://github.com/clguo/SA-UNet) |
| 32 | HREFNet 8方向扫描 | 20.4 | 3.40 | 扫描 | 方向覆盖 | Dynamic Snake + 8 方向 Mamba |
| 33 | MCAttn 蒙特卡洛注意力 | 20.3 | 4.25 | 架构 | 多尺度 | arXiv'26，[GitHub](https://github.com/anthonyweidai/SvANet)，随机多尺度池化 |
| 34 | Hausdorff Distance Loss | 20.3 | 3.28 | 损失 | ③HD95 | [GitHub](https://github.com/PatRyg95/HausdorffLoss)，可微分 HD 近似 |
| 35 | Semi-Mamba-UNet | 20.0 | 4.51 | 训练 | 泛化性 | KBS'24，[GitHub](https://github.com/ziyangwang007/Mamba-UNet)，跨架构蒸馏 |
| 36 | Cosine Annealing LR | 16.3 | 3.28 | 训练 | 训练效率 | 经典 SGDR，1 行代码 |
| 37 | TTA 测试时增强 | 16.3 | 3.28 | 推理 | 预测稳定性 | 多翻转/旋转取平均，Dice +0.5-1% |

> ⚠ 均衡性 < 3.1 的偏科方法——某维度极强但另一维度极弱（通常是创新高但无顶会/源码）。虽有高创新价值，实施风险也高。

---

## 方法详述（按综合得分降序）

### 1. TopoMask 拓扑感知掩码训练（29.3，均衡 4.25）

**论文**：Zhou et al., *"MaskVSC"*, IEEE TMI 2025
**原理**：训练时对 GT 骨架化，随机选取 40% 血管分支用高斯噪声掩码覆盖，迫使模型从上下文推断被遮挡血管。双相调度（0→40%→0），backbone 无关，零推理开销
**针对问题**：②血管断裂——训练时学习血管连通性先验
**集成方式**：在 `dataset_registry.py` 的 `__getitem__` 中添加 TopoMask 变换，~80 行
**组合推荐**：与 Connectivity Loss(4) 同源互补；与 clDice(3) 独立叠加
**预期效果**：clDice +2-3%，血管断裂减少
**学术价值**：backbone 无关的训练策略创新，7 数据集验证
**评分**：创新性 4 | 问题解决力 4 | 工程可行性 3 | 顶会 5 | 契合度 5

### 2. GLCP Loss 图学习连通性保持（29.3，均衡 4.25）

**论文**：MICCAI 2025 Oral（图学习连通性保持损失）
**原理**：通过构建血管图结构，显式优化血管分支的连通性。使用图拓扑而非简单连通域计数
**针对问题**：②血管断裂——比 Connectivity Loss(4) 理论更完善（图结构 vs 连通域计数）
**集成方式**：替代或补充 Connectivity Loss，加入组合损失
**组合推荐**：与 Connectivity Loss(4) 二选一（GLCP 更完善）；与 TopoMask(1) 互补
**预期效果**：拓扑指标显著提升，MICCAI Oral 级别方法
**学术价值**：MICCAI Oral（接受率 <25%），图结构拓扑损失的理论深度高
**评分**：创新性 4 | 问题解决力 4 | 工程可行性 3 | 顶会 5 | 契合度 5

### 3. clDice 损失函数（29.3，均衡 3.80）

**论文**：Shit et al., *"clDice — a Novel Topology-Preserving Loss Function for Tubular Structure Segmentation"*, CVPR 2021
**GitHub**：https://github.com/jocpae/clDice
**原理**：在 Dice Loss 基础上增加中心线 Dice（骨架拓扑一致性），通过可微分骨架化直接优化血管连通性
**针对问题**：②血管断裂/clDice 偏低（CHASE_DB1 0.799 vs SOTA 83.62）
**集成方式**：`(1-α)·DiceBCE + α·(1-soft_clDice)`，α=0.3，替代现有 `0.5·(CE+Dice)`
**组合推荐**：与 Boundary Loss(6)、Connectivity Loss(4) 构成"拓扑+边界+连通"三重损失
**预期效果**：Dice +0.5-1%，HD95 -30%，clDice 显著提升
**学术价值**：消融实验必备对比项（有/无 clDice），直接优化拓扑保持
**评分**：创新性 2 | 问题解决力 5 | 工程可行性 5 | 顶会 5 | 契合度 5

### 4. Connectivity Loss 连通性损失（28.8，均衡 4.25）

**论文**：Zhou et al., *"MaskVSC: Masked Vascular Structure Segmentation and Completion"*, IEEE TMI 2025
**原理**：惩罚预测中连通组件数与 GT 的差异：`L_C = |#C(Y_pred) - #C(Y_gt)| / #C(Y_gt)`，使用总变分近似实现可微分
**针对问题**：②血管断裂——直接惩罚连通域数量差异
**集成方式**：`L = L_BCE + λ·L_C`，λ=1（论文推荐）
**组合推荐**：与 TopoMask(1) 同源（MaskVSC），互补——TopoMask 训练时预防断裂，L_C 损失层面惩罚断裂
**预期效果**：断裂组件数 -30%，clDice 提升
**学术价值**：连通性消融项，实现极简（~30 行）
**评分**：创新性 3 | 问题解决力 4 | 工程可行性 4 | 顶会 5 | 契合度 5

### 5. SvAttn 尺度变异注意力（27.6，均衡 **4.60** ★最均衡）

**论文**：Dai et al., *"Exploiting Scale-Variant Attention for Segmenting Small Medical Objects"*, arXiv/IEEE 2026
**GitHub**：https://github.com/anthonyweidai/SvANet
**原理**：跨尺度注意力机制——在编码器各 stage 间建立注意力映射，追踪细血管在逐步压缩中的变化，高分辨率表示通过跨尺度加权保留
**针对问题**：①细血管在低分辨率 stage 中消失（降采样信息损失）
**集成方式**：在编码器各 stage 间添加跨尺度注意力模块
**组合推荐**：与 MCAttn(33) 同源（SvANet）；与 SerpScan 扫描互补——SerpScan 处理空间连续性，SvAttn 处理尺度连续性
**预期效果**：FIVES 小目标 mDice 85.91%，细血管 Dice +1-2%
**学术价值**：跨尺度特征增强消融项，有开源代码，各维度均衡无短板
**评分**：创新性 4 | 问题解决力 4 | 工程可行性 4 | 顶会 3 | 契合度 4

### 6. Boundary Loss——边界距离损失（27.4，均衡 3.90）

**论文**：Kervadec et al., *"Boundary Loss for Highly Imbalanced Segmentation"*, Medical Image Analysis 2021
**GitHub**：https://github.com/LIVIAETS/boundary-loss
**原理**：基于预计算 GT 符号距离图（SDM），通过积分形式直接惩罚预测边界与真值边界的偏差
**针对问题**：③HD95 过高（FIVES 28.20, CHASE_DB1 12.97，归一化 ~1% 对角线）
**集成方式**：预计算距离图，添加 `λ_b·boundary_loss(pred_softmax, dist_map)` 到现有损失
**组合推荐**：与 Dice Loss 互补（Dice 优化区域重叠，Boundary 优化边界精度）；与 clDice(3) 构成组合损失
**预期效果**：HD95 显著下降，边界精度大幅提升
**学术价值**：边界精度消融项，证明"区域+边界"联合优化的有效性
**评分**：创新性 2 | 问题解决力 5 | 工程可行性 4 | 顶会 5 | 契合度 4

### 7. TAGC 管状感知门控卷积（27.3，均衡 4.25）

**论文**：Shao et al., *"TA-Mamba: Tubular-aware mamba for accurate retinal vessel segmentation"*, Multimedia Systems 2025
**原理**：在 Mamba block 前插入 Dynamic Snake Conv + 1×1 Conv 门控分支，维持 2D→1D 展平前的管状空间关系
**针对问题**：②管状结构保持——TA-Mamba 在 CHASE_DB1 clDice 达 84.48%（SerpMamba 为 79.95%）
**集成方式**：在 `MambaLayer_Serpentine_Scan` 的 SSM 层前插入 TAGC 模块，~150 行
**组合推荐**：与 DFF(25) 同源（TA-Mamba）互补
**预期效果**：clDice +2-4%，CHASE_DB1 有望追平 TA-Mamba 的 84.48%
**学术价值**：Mamba 前置管状感知模块，CHASE_DB1 已验证有效
**评分**：创新性 4 | 问题解决力 4 | 工程可行性 3 | 顶会 3 | 契合度 5

### 8. clCE Loss 中心线交叉熵（26.9，均衡 3.90）

**论文**：MICCAI 2024（无需骨架化的中心线 CE）
**原理**：通过可微分中心线提取（无需显式骨架化），在血管中心线上施加 CE 损失，直接优化细血管检测
**针对问题**：①细血管中心线——无需骨架化即可聚焦中心线像素
**集成方式**：`L = CE + λ·clCE`，与标准 CE 联合训练
**组合推荐**：与 clDice(3) 互补（clCE 聚焦中心线像素 vs clDice 优化拓扑连通性）
**预期效果**：细血管 SE 提升，减少中心线漏检
**学术价值**：无需骨架化算子的中心线损失，MICCAI 级别方法
**评分**：创新性 2 | 问题解决力 4 | 工程可行性 4 | 顶会 5 | 契合度 5

### 9. Boundary DoU Loss——薄壁管状边界损失（26.9，均衡 3.90）

**论文**：*"Boundary Difference Over Union Loss For Medical Image Segmentation"*, MICCAI 2023
**GitHub**：https://github.com/sunfan-bvb/BoundaryDoULoss
**原理**：计算预测与真值在边界区域的差异比（Difference over Union），专门优化薄壁/细管状结构的边界分割
**针对问题**：③HD95 过高 + 细血管边界丢失
**集成方式**：`Loss = α·DiceBCE + β·BoundaryDoU + γ·clDice`，三项分别优化整体、边界、拓扑
**组合推荐**：与 clDice(3)、Boundary Loss(6) 三选一或组合使用
**预期效果**：HD95 下降，Glaucoma 组 SE 提升
**学术价值**：细血管边界消融项
**评分**：创新性 2 | 问题解决力 4 | 工程可行性 5 | 顶会 5 | 契合度 4

### 10. 注意力门控跳跃连接（26.9，均衡 3.90）

**论文**：Schlemper et al., *"Attention-Gated Networks for Medical Image Segmentation"*, MICCAI 2019
**GitHub**：https://github.com/ozan-oktay/Attention-Gated-Network
**原理**：使用解码器门控信号生成注意力图，过滤编码器跳跃特征中的背景噪声（渗出、出血等），仅让血管相关特征通过
**针对问题**：①②解码器背景噪声干扰（SerpMamba 使用简单 `torch.cat` 拼接跳跃连接）
**集成方式**：在 5 级跳跃连接中插入 `AttentionGate(F_g, F_l, F_int)`，替换 `torch.cat`
**组合推荐**：与 SvAttn(5) 不互斥；与 CoordAtt(16) 可并行
**预期效果**：Dice +0.5-1%，AMD/DR 组 SP 提升（减少假阳性）
**学术价值**：经典跳跃连接增强消融项
**评分**：创新性 2 | 问题解决力 4 | 工程可行性 5 | 顶会 5 | 契合度 4

### 11. FcaNet 频域通道注意力（26.3，均衡 4.25）

**论文**：Qin et al., *"FcaNet: Frequency Channel Attention Networks"*, ICCV 2021
**GitHub**：https://github.com/cfzd/FcaNet
**原理**：用 DCT 频率分量替代 SE 的全局平均池化，使网络关注不同频率子带（细血管=高频，粗血管=低频）
**针对问题**：③频域增强——与 C7 小波增强互补（FcaNet 通道维度，C7 空间维度）
**集成方式**：替代编码器中的 SE 模块（代码已有 `squeeze_excitation` 标志），或独立插入
**组合推荐**：与 WaveRNet(15) 互补（DCT 通道 vs DWT 空间）；可替换 SE 模块
**预期效果**：频域特征增强，血管对比度改善
**学术价值**：频域注意力消融项，DCT+Mamba 是新组合
**评分**：创新性 3 | 问题解决力 3 | 工程可行性 4 | 顶会 5 | 契合度 4

### 12. 细血管加权 BCE Loss（26.3，均衡 3.83）

**论文**：Linaris et al., *"Vessel-Width-Based Metrics and Weight Masks for Retinal Blood Vessel Segmentation"*, SIBGRAPI 2024
**GitHub**：https://github.com/J-Linaris/retinal_thin_vessels
**原理**：基于距离变换生成权重图，距离血管中心线越远的像素（细血管边界）权重越高，使 BCE 关注细血管
**针对问题**：①细血管检测（FIVES Glaucoma Dice 0.814，含灾难性失败）+ ④模型保守（SE < PR）
**集成方式**：预计算或在线计算权重掩码，加权 BCE 替换标准 BCE
**组合推荐**：与 AG-TAL(19) 互补（距离变换加权 vs 半径感知加权）；与 clDice(3) 联合
**预期效果**：SE +3-5%（细血管区域），Glaucoma 样本 Dice 显著改善
**学术价值**：细血管专项消融项
**评分**：创新性 2 | 问题解决力 5 | 工程可行性 4 | 顶会 3 | 契合度 5

### 13. EfficientVMamba 空洞扫描策略（26.3，均衡 3.83）

**论文**：Pei et al., *"EfficientVMamba: Atrous-Based Selective Scanning for Efficient Visual SSM"*, AAAI 2025
**原理**：将空洞卷积思想引入 Mamba 扫描——按固定间隔跳过像素获得更大感受野，可学习的扫描间隔自适应调整采样密度
**针对问题**：①多尺度——与 SerpScan 的蛇形扫描互补，在方向维度增加尺度维度
**集成方式**：→C6 空洞蛇形扫描的灵感来源，在 SerpScan 中引入 dilation 参数
**组合推荐**：→与拓扑感知蛇形扫描(18)互补——C2 解决"沿着哪里扫"，空洞扫描解决"以多粗粒度扫"
**预期效果**：→细血管 Dice +2-3%，计算量增加 <15%
**学术价值**：→空洞蛇形扫描(C6) 是全新 SSM 扫描策略，理论+实验空间大
**评分**：创新性 4 | 问题解决力 3 | 工程可行性 2 | 顶会 5 | 契合度 5

### 14. ADDR 阈值自适应（26.0，均衡 3.04 ⚠偏科）★原创

**原理**：将 ADDR 模块的固定阈值改为输入条件化的自适应阈值——通过轻量网络根据输入特征动态预测阈值，使模型在异常样本上也能做出合理分割
**针对问题**：①灾难性失败——FIVES 含 2 张 Glaucoma Dice<0.30 样本，固定阈值是根因之一
**集成方式**：在 ADDR 模块中增加阈值预测分支，~100 行
**组合推荐**：与病种感知 ADDR(24) 互补；可与 InTEnt(17) 叠加（训练时+推理时双重安全网）
**预期效果**：灾难性失败样本 Dice 0.13→>0.70，整体 Dice +1-2%
**学术价值**：**核心创新贡献**——自适应双校准阈值，CVPR/MICCAI 竞争力。⚠均衡性低：创新高但无顶会/源码
**评分**：创新性 5 | 问题解决力 5 | 工程可行性 3 | 顶会 0 | 契合度 5

### 15. WaveRNet 小波注意力（25.7，均衡 4.20）

**论文**：Wang et al., *"WaveRNet: Wavelet-Based Attention Module for Domain-Generalizable Retinal Vessel Segmentation"*, ISBI 2024
**GitHub**：https://github.com/Chanchan-Wang/WaveRNet
**原理**：DWT 分解特征为 LL/LH/HL/HH 四子带，分别施加注意力——LL 增强全局血管结构，HH 增强高频边缘
**针对问题**：③低质量图像鲁棒性——频域多尺度特征增强，AMD/DR 低对比度图像受益
**集成方式**：在 `MambaLayer_Serpentine_Scan` 后插入 `WaveletAttention`，~100 行
**组合推荐**：与 FcaNet(11) 互补（DWT 空间域 vs DCT 通道域）；→可延伸为频域增强蛇形扫描(23)
**预期效果**：低质量图像 Dice +3-5%，HD95 下降
**学术价值**：小波+Mamba SSM 是新颖交叉方向
**评分**：创新性 3 | 问题解决力 4 | 工程可行性 3 | 顶会 3 | 契合度 5

### 16-37. 其他方法（简述）

| # | 方法 | 总分 | 均衡 | 关键信息 |
|---|------|------|------|---------|
| 16 | **CoordAtt** | 25.3 | 3.83 | CVPR'21，[GitHub](https://github.com/Andrew-Qibin/CoordAttention)，x/y方向编码。创新性2\|问题解决力3\|可行性5\|顶会5\|契合4 |
| 17 | **InTEnt** | 25.0 | 3.98 | NeurIPS'23，[GitHub](https://github.com/mazurowski-lab/single-image-test-time-adaptation)，单图 BN 微调。创新性2\|问题解决力4\|可行性4\|顶会5\|契合3 |
| 18 | **拓扑感知蛇形扫描** ★ | 24.9 | 2.94⚠ | 原创：拓扑先验引导 SerpScan。创新性5\|问题解决力5\|可行性2\|顶会0\|契合5 |
| 19 | **AG-TAL** | 24.6 | 3.14 | 半径感知 Dice + 断裂感知 clDice。创新性4\|问题解决力5\|可行性3\|顶会0\|契合5 |
| 20 | **DCNv4** | 24.0 | 3.98 | CVPR'24，[GitHub](https://github.com/OpenGVLab/DCNv4)，3×加速可变形卷积。创新性3\|问题解决力3\|可行性2\|顶会5\|契合4 |
| 21 | **Focal Tversky** | 24.0 | 3.98 | ISBI'19，[GitHub](https://github.com/nabsabraham/focal-tversky-unet)，可调 FN 惩罚。创新性2\|问题解决力4\|可行性5\|顶会3\|契合3 |
| 22 | **ASPP** | 24.0 | 3.50 | ECCV'18 经典，多速率并行空洞卷积。创新性1\|问题解决力4\|可行性5\|顶会5\|契合3 |
| 23 | **频域增强蛇形扫描** ★ | 23.6 | 3.06⚠ | 原创：DWT 子带引导 SerpScan。创新性5\|问题解决力4\|可行性2\|顶会0\|契合5 |
| 24 | **病种感知 ADDR** ★ | 23.3 | 3.28 | 原创：病种嵌入条件化阈值。创新性4\|问题解决力4\|可行性3\|顶会0\|契合5 |
| 25 | **DFF** | 22.8 | 4.25 | TA-Mamba(Shao'25)，方向权重调制跳跃连接。创新性2\|问题解决力3\|可行性4\|顶会3\|契合4 |
| 26 | **DPGNet** | 22.4 | 4.37 | IEEE JBHI'25，边界差异引导注意力。创新性2\|问题解决力4\|可行性3\|顶会3\|契合3 |
| 27 | **WTCM-UNet** | 22.4 | 4.37 | Signal Processing'26，小波系数注入 Mamba。创新性3\|问题解决力3\|可行性2\|顶会3\|契合4 |
| 28 | **CLAHE** | 22.2 | 3.67 | berenslab 验证 Dice +2-3%，1 行代码。创新性1\|问题解决力4\|可行性5\|顶会3\|契合3 |
| 29 | **HMS-VesselNet** | 21.0 | 3.53 | 难例挖掘 + clDice 联合。创新性3\|问题解决力4\|可行性3\|顶会0\|契合4 |
| 30 | **Polygon-Mamba** | 20.8 | 3.28 | PS-VSS 多边形扫描+SFCAM。创新性4\|问题解决力3\|可行性2\|顶会0\|契合5 |
| 31 | **SA-UNet** | 20.7 | 3.74 | Applied Intelligence'21，[GitHub](https://github.com/clguo/SA-UNet)。创新性1\|问题解决力3\|可行性5\|顶会3\|契合3 |
| 32 | **HREFNet** | 20.4 | 3.40 | Dynamic Snake+8方向 Mamba。创新性3\|问题解决力3\|可行性3\|顶会0\|契合5 |
| 33 | **MCAttn** | 20.3 | 4.25 | SvANet(Dai'26)，[GitHub](https://github.com/anthonyweidai/SvANet)，随机多尺度池化。创新性2\|问题解决力2\|可行性4\|顶会3\|契合3 |
| 34 | **HD Loss** | 20.3 | 3.28 | [GitHub](https://github.com/PatRyg95/HausdorffLoss)，可微分 HD 近似。创新性2\|问题解决力4\|可行性5\|顶会0\|契合3 |
| 35 | **Semi-Mamba-UNet** | 20.0 | 4.51 | KBS'24，[GitHub](https://github.com/ziyangwang007/Mamba-UNet)，跨架构蒸馏。创新性3\|问题解决力2\|可行性2\|顶会3\|契合3 |
| 36 | **Cosine Annealing** | 16.3 | 3.28 | 经典 SGDR，1行代码。创新性1\|问题解决力3\|可行性5\|顶会0\|契合2 |
| 37 | **TTA** | 16.3 | 3.28 | 多翻转/旋转推理取平均。创新性1\|问题解决力3\|可行性5\|顶会0\|契合2 |

---

## 推荐组合方案

### 方案 A：工程基线（1-2 天，不改模型）

| 方法 | 解决问题 | 成本 |
|------|---------|------|
| CLAHE 预处理(28) | 全局基础 | 1 行 |
| Cosine Annealing LR(36) | 训练效率 | 1 行 |
| TTA(37) | 预测稳定性 | ~30 行 |

预期：FIVES Dice 0.85→0.87-0.88，CHASE_DB1 0.81→0.83-0.84

### 方案 B：拓扑+边界+细血管专项（3-5 天，推荐首选）

| 方法 | 均衡性 | 解决问题 | 成本 |
|------|--------|---------|------|
| 方案 A 全部 | — | 基线 | — |
| TopoMask(1) | 4.25 | 血管断裂 | ~80 行 |
| clDice(3) | 3.80 | 拓扑连通 | ~50 行 |
| Boundary Loss(6) | 3.90 | HD95 过高 | ~40 行 |
| 细血管加权 BCE(12) | 3.83 | 细血管 | ~50 行 |
| SvAttn(5) ★最均衡 | 4.60 | 细血管消失 | ~100 行 |

组合损失：`L = 0.3·CE + 0.2·Dice + 0.2·clDice + 0.15·Boundary + 0.15·Connectivity`
预期：clDice +3-5%，HD95 -30%，Glaucoma 灾难性失败大幅减少

### 方案 C：论文贡献（4-6 周）

| 贡献点 | 对应方法 | 创新性 | 均衡性 |
|--------|---------|--------|--------|
| **核心贡献** | 拓扑感知蛇形扫描(18) + 空洞蛇形扫描(13→C6) | 5 | ⚠偏科 |
| **贡献2** | ADDR 阈值自适应(14) | 5 | ⚠偏科 |
| **均衡增强** | SvAttn(5) ★最均衡 | 4 | 4.60 |
| 实验基线 | 方案 A + B | 工程调优 | — |
| 消融对比 | clDice/Boundary Loss/Focal Tversky 等损失对比 | 补充实验 | — |

目标：MICCAI 2026 oral / IEEE TMI
