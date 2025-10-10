# AlzheimerDl 🧠

一个专注阿尔茨海默病（Alzheimer's Disease）MRI 影像自动分类/辅助诊断的深度学习实验工程。核心目标：在保持结构清晰与可扩展性的同时，快速迭代模型（ResNet 3D 预训练 + Mosa 自注意力结构），并提供完整的训练指标与 TensorBoard 可视化支持。

> 这个仓库不是“巨石工程”，而是一个便于实验、快速对比、稳健恢复训练的轻量化研究代码基。欢迎在其上扩展新模型或进行数据增强实验。

---

## ✨ 主要特性

- 3D ResNet 预训练与评估脚本（`pretrain_resnet.py`）
- Mosa 结构网络训练脚本（`train_mosa.py`），支持共享 Transformer 与多头注意力调优
- Focal Loss / Cross Entropy 可切换，支持 Label Smoothing
- 多指标记录：Loss、Accuracy、Recall、F1 (weighted / macro)、AUC、Precision、Specificity
- 断点恢复支持（checkpoint + TensorBoard 连续 `global_step`，避免曲线错位）
- 自动保存最佳模型（`model_best.pth.tar`）与周期性 checkpoint
- 直方图 & 混淆矩阵 & ROC/PR 曲线 & per-class metrics JSON 导出
- 训练结束生成 `summary_all.json` 汇总最佳验证与测试指标
- 结构化配置（dataclass），方便修改超参数与日志行为

---

## 📂 目录结构

```text
augment.py              # 数据增强（如需扩展可在此集中处理）
config.py               # 全局配置（数据 / 模型 / 训练 / 日志）
data_loader.py          # 数据集与 DataLoader 构造
environment.yml         # Conda 环境（完整版依赖）
metrics_utils.py        # 指标计算与绘图辅助
model_builder.py        # 模型构造入口（根据配置构建）
mosa_net.py / mosa_net_simple.py  # Mosa 网络实现
pretrain_resnet.py      # 3D ResNet 预训练与评估脚本
resnet.py               # 3D ResNet 实现
train_mosa.py           # Mosa 网络训练主脚本
utils.py                # 通用工具（checkpoint、accuracy等）
requirements.txt        # 精简 pip 依赖
README.md               # 项目说明（当前文件）
```

---

## 🛠 环境准备

推荐使用 Conda：

```bash
conda env create -f environment.yml
conda activate base   # 或修改 environment.yml 中的 name
```

若仅快速运行核心训练：

```bash
pip install -r requirements.txt
```

依赖核心：`torch`、`torchio`、`numpy`、`scikit-learn`、`matplotlib`、`seaborn`、`tqdm`、`einops`。

> CUDA 版本：脚本当前使用 `torch==2.5.0+cu124`，请确保本地驱动与 CUDA 兼容。

---

## 📁 数据说明

在 `config.py` 中通过 `DataConfig.data_root` 指定数据根目录：

```python
data_root: str = "/root/01_dataset/ADNI_dataset_compressed"
```

数据加载逻辑在 `data_loader.py`，需保证结构与读取逻辑匹配（你可以在此扩展自定义数据拆分、缓存、预处理）。

基本假设：MRI 体数据 + 二分类标签（AD vs Control）。若扩展多分类（MCI 等），需同步调整 `BackboneConfig.num_classes`。

---

## 🚀 快速开始

预训练 3D ResNet：

```bash
python pretrain_resnet.py
```

训练 Mosa 网络：

```bash
python train_mosa.py
```

修改超参数：直接编辑 `config.py` 或在脚本中实例化后覆写：

```python
from config import Config
cfg = Config()
cfg.training.learning_rate = 5e-5
cfg.backbone.model_type = "resnet50_3d"
```

---

## 🔄 恢复训练

将某次保存的 checkpoint 路径赋值给 `TrainingConfig.resume`：

```python
cfg.training.resume = "./checkpoints/checkpoint.pth.tar"
```

恢复时：

- 自动加载模型 / 优化器 / 调度器状态
- 复用原 `log_dir`
- `global_step` 连续递增（避免 TensorBoard 曲线回退）
- `purge_step` 清理未完成写入的尾部事件（只在恢复时启用）

---

## 📊 指标与可视化

TensorBoard：

```bash
tensorboard --logdir=./logs
```

记录内容分类：

- `Train/Batch/*`：批级指标（可根据需要裁剪）
- `Train/*`, `Val/*`, `Test/*`：epoch 级指标与图像
- `Params/*`, `Grads/*`：直方图（由 `LoggingConfig.hist_interval` 控制间隔）
- `summary_all.json`：训练结束综合指标（最佳验证 + 测试）
- `best_val_per_class_metrics.json` / `test_per_class_metrics.json`：每类统计与支持度

ROC/PR：仅在二分类时记录；多分类可自行扩展 OvR/宏平均。

---

## 🧪 模型与超参数

核心可调：

| 类 | 关键字段 | 说明 |
|----|----------|------|
| `BackboneConfig` | `model_type` | `resnet18_3d` / `resnet50_3d` |
| `MosaConfig` | `embed_dim`, `depth`, `num_heads` | 控制注意力块容量 |
| `TrainingConfig` | `optimizer`, `learning_rate`, `lr_scheduler` | 支持 adam / adamw / sgd；cosine / step / plateau |
| `TrainingConfig` | `loss_fn` | `cross_entropy` 或 `focal` |
| `LoggingConfig` | `hist_interval` | 直方图记录间隔（epoch） |

> 使用 Focal Loss 时建议学习率稍低；多分类扩展时别忘记同步更新 `num_classes` 与标签处理。

---

## 🧩 断点恢复与 step 连续性实现细节

- checkpoint 中保存：`epoch`、`best_acc`、`optimizer`、`scheduler`、`log_dir`、`global_step`
- 恢复后：
  - `config.training.start_epoch` 和 `config.training.prev_global_step` 注入配置
  - 每个 batch 的 `global_step = prev_global_step + (epoch - start_epoch) * len(train_loader) + batch_idx`
  - TensorBoard 使用 `purge_step` 避免残留事件覆盖

---

## 📦 Checkpoints

默认目录：`./checkpoints`

文件：

- `checkpoint.pth.tar`（最近一次保存）
- `model_best.pth.tar`（验证集最佳）

---

## ✅ 常见问题（FAQ）

**Q: 运行时报 CUDA OOM？**  
A: 降低 `DataConfig.batch_size` 或减少 `MosaConfig.embed_dim`。

**Q: 日志太大？**  
A: 关闭 batch 级写入或调高 `LoggingConfig.hist_interval`。

**Q: 曲线断点后错位？**  
A: 确认恢复时未更改 `batch_size`；否则建议新开 log 目录。

**Q: 指标不稳定波动很大？**  
A: 关注 `Train/Batch/*` 是否太嘈杂，可仅保留 epoch 聚合指标。

---

## 🗺 后续潜在改进（Roadmap）

- 分布式训练（DDP）与多 GPU 支持
- 自动学习率搜索（OneCycle / Warmup + Cosine 更多策略）
- 更丰富的数据增强（时空 CutMix / Mixup）
- 多分类任务（AD / MCI / CN）拓展与类不均衡重加权
- 模型结构：加入 Swin / ConvNeXt 3D / ViT 3D 对比
- 半监督 / 伪标签机制集成

---

## 🤝 贡献

欢迎 Issue 与 PR：

- 性能优化
- 新指标可视化
- 代码结构简化 / 模块解耦

---

## 📜 许可证

当前未指定 License，若需开放发布建议添加 `LICENSE`（MIT/Apache-2.0）。

---

## 🧪 快速测试（示例）

```bash
python pretrain_resnet.py --help  # 如果后续添加 argparse
python pretrain_resnet.py         # 默认配置跑预训练
python train_mosa.py              # 训练 Mosa 模型
```

---

## 🧬 作者寄语

这个工程希望把“MRI 分类实验”从凌乱脚本变成结构化、可恢复、易复用的代码基。你的改动也应该保持：清晰命名、最小副作用、日志可追踪。持续迭代，稳扎稳打。欢迎一起探索更好的阿尔茨海默病智能分析方法。

---

如果有问题或需要新增功能，欢迎直接发起讨论。祝实验顺利！
