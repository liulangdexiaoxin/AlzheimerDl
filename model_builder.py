import torch
import torch.nn as nn
from resnet import resnet18_3d, resnet50_3d
from mosa_net_simple import MosaShard
from mosa_net import Mosa

def build_backbone(config):
    """构建ResNet backbone"""
    if config.backbone.model_type == "resnet18_3d":
        model = resnet18_3d(
            num_classes=config.backbone.num_classes,
            in_channels=config.backbone.in_channels
        )
    elif config.backbone.model_type == "resnet50_3d":
        model = resnet50_3d(
            num_classes=config.backbone.num_classes,
            in_channels=config.backbone.in_channels
        )
    else:
        raise ValueError(f"Unsupported model type: {config.backbone.model_type}")
    
    # 加载预训练权重
    if config.backbone.pretrained and config.backbone.pretrained_path:
        print(f"Loading pretrained weights from {config.backbone.pretrained_path}")
        checkpoint = torch.load(config.backbone.pretrained_path, map_location='cpu')
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
            
        # 移除可能存在的模块前缀
        for k in list(state_dict.keys()):
            if k.startswith('module.'):
                state_dict[k[7:]] = state_dict.pop(k)
                
        # 加载权重
        model.load_state_dict(state_dict, strict=False)
        
    return model

def build_mosa_net(config):
    """构建完整的MosaNet模型"""
    backbone = build_backbone(config)
    
    if config.mosa.use_shared_transformer:
        model = MosaShard(
            backbone=backbone,
            num_classes=config.backbone.num_classes,
            embed_dim=config.mosa.embed_dim,
            depth=config.mosa.depth,
            num_heads=config.mosa.num_heads
        )
    else:
        model = Mosa(
            backbone=backbone,
            num_classes=config.backbone.num_classes,
            embed_dim=config.mosa.embed_dim,
            depth=config.mosa.depth,
            num_heads=config.mosa.num_heads
        )
    
    return model

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)