# 多模态MRD状态预测项目计划 (MRD Multimodal Prediction)

## 1. 项目背景与目标
**目标**：构建一个多模态神经网络，用于预测MRD (微小残留病灶) 状态 (0/1分类)。
**输入模态**：
1. **影像数据**：4个序列的3D CT图像 (.nii格式)。
2. **临床数据**：患者的临床信息 (如年龄、性别、病理特征、基因突变等)。
**核心架构设计**：
- **CT特征提取层**：使用预训练的3D CNN提取图像特征，且在训练过程中**冻结权重 (不finetune)**。
- **临床特征提取层**：使用全连接网络 (MLP) 编码临床表格数据。
- **多模态融合层 (Fusion)**：参考CPnet (Cross-Partial Network或其他跨模态机制)，将CT的高维特征与临床特征进行有效融合。
- **预测层**：基于融合后的特征输出MRD状态的概率 (Binary Classification)。

## 2. 目录结构与模块说明
已经在项目中建立了如下目录结构：
```text
mrd_multimodal_prediction/
├── data/                       # 数据存放目录
│   ├── raw/                    # 存放原始的 .nii 影像和临床 csv/excel 文件
│   └── processed/              # 存放预处理后的数据 (如重采样、裁剪后的CT矩阵)
├── src/                        # 核心代码模块
│   ├── data/                   # 数据读取与处理模块
│   │   ├── dataset.py          # PyTorch Dataset 定义，负责读取4序列CT和临床数据
│   │   ├── transforms.py       # 3D 图像的预处理与数据增强 (如缩放、裁剪、归一化)
│   │   └── clinical_processor.py # 临床表格数据的预处理 (如缺失值填补、归一化、独热编码)
│   ├── models/                 # 模型架构模块
│   │   ├── ct_extractor.py     # 3D CT 预训练特征提取器 (权重冻结)
│   │   ├── clinical_encoder.py # 临床特征编码器 (MLP)
│   │   ├── fusion.py           # 跨模态融合层 (参考CPnet实现)
│   │   └── mrd_predictor.py    # 最终的顶层模型，拼装上述所有模块
│   ├── training/               # 训练核心逻辑
│   │   ├── trainer.py          # 训练循环、验证循环实现
│   │   ├── losses.py           # 损失函数 (如 BCEWithLogitsLoss, Focal Loss)
│   │   └── metrics.py          # 评价指标 (AUC, Accuracy, Sensitivity, Specificity)
│   └── utils/                  # 辅助工具
│       ├── config.py           # 读取 default.yaml 配置
│       └── logger.py           # 日志记录 (TensorBoard / wandb集成)
├── configs/                    # 配置文件目录
│   └── default.yaml            # 超参数及路径配置
├── scripts/                    # 执行脚本
│   ├── preprocess.py           # 离线数据预处理脚本
│   ├── train.py                # 启动训练主脚本
│   └── evaluate.py             # 模型测试与评估脚本
├── notebooks/                  # 实验与探索
│   └── eda.ipynb               # 数据探索性分析 (CT可视化、临床数据分布)
├── requirements.txt            # 项目依赖 (torch, monai, nibabel, pandas等)
└── README.md                   # 项目说明
└── PLAN.md                     # 本计划文档
```

## 3. 开发阶段与里程碑 (Milestones)

### 阶段一：数据准备与预处理 (Data Preparation)
- **任务1**：收集并整理4个序列的3D CT (.nii) 以及对应的临床数据表格。
- **任务2**：编写 `clinical_processor.py`，清洗临床数据，提取结构化特征。
- **任务3**：编写 `transforms.py` 和 `dataset.py`。建议使用 **MONAI** 库来处理 `.nii` 图像，实现3D重采样(Resample)、裁剪(Crop)、归一化(Normalize)等操作，并确保4个序列能够堆叠为 `(C, D, H, W)` 形状的Tensor，其中 C=4。

### 阶段二：模型构建 (Model Architecture)
- **任务1**：在 `ct_extractor.py` 中引入预训练的3D CNN (例如MedicalNet预训练权重)。设置 `requires_grad = False` 冻结参数。
- **任务2**：在 `clinical_encoder.py` 中构建多层感知机(MLP)。
- **任务3**：在 `fusion.py` 中实现参考 CPnet 的多模态融合机制。
- **任务4**：在 `mrd_predictor.py` 组装特征提取器、编码器、融合层及最终的分类头 (Classifier)。

### 阶段三：训练与评估框架搭建 (Training & Evaluation)
- **任务1**：编写 `losses.py`，因为MRD预测通常面临正负样本不平衡的问题，建议实现 Focal Loss 或者加权 BCE Loss。
- **任务2**：编写 `trainer.py` 和 `train.py`，实现标准的训练与验证循环。
- **任务3**：编写 `metrics.py`，加入 ROC-AUC、PR-AUC 等医学临床关心的指标计算。

### 阶段四：实验与调优 (Experiment & Tuning)
- 运行 `train.py` 进行初步训练。
- 由于CT特征提取层被冻结，主要训练压力在 Fusion 层和 Clinical Encoder，训练速度会较快。
- 调整学习率、批次大小以及融合层的复杂度，观察验证集指标。

## 4. 推荐的技术栈
- **深度学习框架**：PyTorch
- **医学图像处理框架**：MONAI (提供成熟的医学图像Dataset、Transforms和3D网络组件)
- **图像I/O库**：nibabel, SimpleITK
- **数据处理**：pandas, numpy, scikit-learn
- **实验追踪**：Weights & Biases (wandb) 或 TensorBoard
