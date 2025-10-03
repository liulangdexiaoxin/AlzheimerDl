import torch
from config import Config
from data_loader import get_data_loaders
from model_builder import build_backbone, build_mosa_net
from utils import load_checkpoint, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import numpy as np
import os
from tqdm import tqdm

def test_model(model, test_loader, config, class_names=['CN', 'AD']):
    """测试模型性能"""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target, _ in pbar:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    if len(class_names) == 2:
        auc = roc_auc_score(all_targets, [p[1] for p in all_probs])
    else:
        auc = roc_auc_score(all_targets, all_probs, multi_class='ovr')
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(config.training.log_dir, 'confusion_matrix.png'))
    weighted_f1 = report['weighted avg']['f1-score']
    print(f"\nWeighted F1-score: {weighted_f1:.4f}")
    print("\n" + "="*50)
    print("测试集性能评估")
    print("="*50)
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print("\n分类报告:")
    print(classification_report(all_targets, all_preds, target_names=class_names))
    print("\n混淆矩阵:")
    print(cm)
    return test_acc, auc, cm, report

def main():
    # 加载配置
    config = Config()
    config.backbone.pretrained = True
    config.backbone.pretrained_path = "/path/to/resnet_pretrained_weights.pth"  # 替换为ResNet预训练权重路径
    config.training.resume = "/path/to/mosa_simple_trained_weights.pth"  # 替换为MosaSimple训练权重路径

    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)

    # 构建模型
    model = build_mosa_net(config)
    model = model.to(config.device)

    # 加载MosaSimple训练权重
    if config.training.resume:
        print(f"加载MosaSimple训练权重: {config.training.resume}")
        load_checkpoint(config.training.resume, model)

    # 测试模型
    print("开始测试模型...")
    test_acc, auc, cm, report = test_model(model, test_loader, config)

    print(f"\n测试完成，结果如下:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()