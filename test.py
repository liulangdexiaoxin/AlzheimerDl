# TODO: 完善测试脚本，添加更多评估指标和可视化功能
import os
import torch
import numpy as np
from tqdm import tqdm
import json

from config import Config
from data_loader import get_data_loaders
from model_builder import build_mosa_net
from utils import load_checkpoint, evaluate_metrics, plot_confusion_matrix

def test(model, test_loader, config):
    """测试模型"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for data, target, _ in tqdm(test_loader, desc='Testing'):
            data, target = data.to(config.device), target.to(config.device)
            
            # 前向传播
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    return all_targets, all_preds, all_probs

def main():
    # 加载配置
    config = Config()
    config.backbone.pretrained = True
    config.backbone.pretrained_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    
    # 创建模型
    model = build_mosa_net(config)
    model = model.to(config.device)
    
    # 加载最佳模型
    best_model_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        load_checkpoint(best_model_path, model)
        print(f"Loaded best model from {best_model_path}")
    else:
        print("No best model found. Using randomly initialized model.")
    
    # 创建数据加载器
    _, _, test_loader = get_data_loaders(config)
    
    # 测试模型
    targets, preds, probs = test(model, test_loader, config)
    
    # 计算评估指标
    class_names = ['CN', 'AD']  # 根据你的数据集修改
    cm, report, auc = evaluate_metrics(targets, preds, probs, class_names)
    
    # 打印结果
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=class_names))
    print(f"AUC: {auc:.4f}")
    
    # 绘制混淆矩阵
    plot_confusion_matrix(cm, class_names, 
                         save_path=os.path.join(config.training.log_dir, 'confusion_matrix.png'))
    
    # 保存结果
    results = {
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'auc': auc
    }
    
    with open(os.path.join(config.training.log_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    main()