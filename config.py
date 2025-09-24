import os
import torch
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataConfig:
    data_root: str = "/root/01_dataset/ADNI_dataset_compressed"
    # data_root: str = "/root/01_dataset/AD_compressed/ADNI_dataset"
    batch_size: int = 24    # 根据独立显存大小设置，太大会导致每个epoch时间暴增，使用Tesla P40 24GB显存
    num_workers: int = 6    # 根据CPU核心数和IO能力设置（CPU 6核，12GB内存）
    target_size: int = 128
    # train_split: float = 0.7
    # val_split: float = 0.15
    # test_split: float = 0.15

@dataclass
class BackboneConfig:
    model_type: str = "resnet18_3d"  # "resnet18_3d" or "resnet50_3d"
    pretrained: bool = False
    pretrained_path: Optional[str] = None
    in_channels: int = 1
    num_classes: int = 2

@dataclass
class MosaConfig:
    """
    MosaConfig类用于存储Mosa模型的配置参数
    包含模型结构、注意力机制、dropout率等关键超参数
    """
    use_shared_transformer: bool = True  # 使用共享权重版本
    embed_dim: int = 256
    depth: int = 2
    num_heads: int = 8
    mlp_ratio: float = 4.0
    qkv_bias: bool = False
    drop_rate: float = 0.1
    attn_drop_rate: float = 0.1

@dataclass
class TrainingConfig:

    """
    训练配置类，包含模型训练所需的各种参数设置。
    包括优化器、学习率调度、训练参数、损失函数以及检查点与日志相关配置。
    """
    # 优化器相关参数
    optimizer: str = "adamw"  # "adam", "adamw", "sgd"
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9  # 用于SGD
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau"
    step_size: int = 10  # 用于step scheduler
    gamma: float = 0.1  # 用于step scheduler
    min_lr: float = 1e-6  # 用于cosine/plateau scheduler
    patience: int = 5  # 用于plateau scheduler
    
    # 训练参数
    num_epochs: int = 100
    warmup_epochs: int = 5
    gradient_clip: float = 1.0
    
    # 损失函数
    loss_fn: str = "focal"  # "cross_entropy", "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    label_smoothing: float = 0.1
    
    # 检查点与日志
    checkpoint_dir: str = "./checkpoints"
    log_dir: str = "./logs"
    save_freq: int = 10  # 每多少epoch保存一次
    resume: Optional[str] = None  # 恢复训练的检查点路径

@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    mosa: MosaConfig = field(default_factory=MosaConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    
    def __post_init__(self):
        # 创建必要的目录
        os.makedirs(self.training.checkpoint_dir, exist_ok=True)
        os.makedirs(self.training.log_dir, exist_ok=True)