import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
import numpy as np
import os
import time
import datetime
import json
from tqdm import tqdm
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

from config import Config
from data_loader import get_data_loaders
from model_builder import build_mosa_net, count_parameters
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, FocalLoss, plot_confusion_matrix

def create_optimizer(model, config):
    """创建优化器"""
    if config.training.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")
    
    return optimizer

def create_scheduler(optimizer, config, train_loader):
    """创建学习率调度器"""
    if config.training.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.training.num_epochs * len(train_loader),
            eta_min=config.training.min_lr
        )
    elif config.training.lr_scheduler == "step":
        scheduler = StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif config.training.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',
            patience=config.training.patience,
            factor=config.training.gamma,
            min_lr=config.training.min_lr
        )
    else:
        scheduler = None
    
    return scheduler

def create_loss_fn(config):
    """创建损失函数"""
    if config.training.loss_fn == "cross_entropy":
        criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    elif config.training.loss_fn == "focal":
        criterion = FocalLoss(
            alpha=config.training.focal_alpha,
            gamma=config.training.focal_gamma
        )
    else:
        raise ValueError(f"Unsupported loss function: {config.training.loss_fn}")
    
    return criterion

def train_epoch(model, train_loader, optimizer, criterion, scheduler, epoch, config):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    batch_losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.training.num_epochs} [Train]')
    for batch_idx, (data, target, _) in enumerate(pbar):
        data, target = data.to(config.device), target.to(config.device)
        output = model(data)
        loss = criterion(output, target)
        batch_losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        if config.training.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
        optimizer.step()
        if scheduler and config.training.lr_scheduler == "cosine":
            scheduler.step()
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        preds = torch.argmax(output, dim=1)
        recall = recall_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
        recalls.update(recall, data.size(0))
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'Acc': f'{top1.avg:.2f}%',
            'Recall': f'{recalls.avg:.2f}'
        })
    auc = roc_auc_score(all_targets, all_preds)
    return losses.avg, top1.avg, recalls.avg, auc, batch_losses, all_targets, all_preds

def validate(model, val_loader, criterion, config):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for data, target, _ in pbar:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            preds = torch.argmax(output, dim=1)
            recall = recall_score(target.cpu().numpy(), preds.cpu().numpy(), average='macro', zero_division=0)
            recalls.update(recall, data.size(0))
            all_targets.extend(target.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'Acc': f'{top1.avg:.2f}%',
                'Recall': f'{recalls.avg:.2f}'
            })
    auc = roc_auc_score(all_targets, all_preds)
    return losses.avg, top1.avg, recalls.avg, auc, all_targets, all_preds

def plot_learning_curves(train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制并保存学习曲线"""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, label='Train Acc')
    plt.plot(epochs, val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'learning_curves.png'))
    plt.close()

def plot_full_learning_curves(
    train_batch_losses, train_epoch_losses, val_epoch_losses, val_epoch_accs, save_dir
):
    """绘制并保存详细学习曲线"""
    steps_batch = range(1, len(train_batch_losses) + 1)
    steps_epoch = range(1, len(train_epoch_losses) + 1)
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 2, 1)
    plt.plot(steps_batch, train_batch_losses, label='Train/Batch Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Train/Batch Loss')
    plt.legend()
    plt.subplot(2, 2, 2)
    plt.plot(steps_epoch, train_epoch_losses, label='Train/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Epoch Loss')
    plt.legend()
    plt.subplot(2, 2, 3)
    plt.plot(steps_epoch, val_epoch_losses, label='Val/Epoch Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Val/Epoch Loss')
    plt.legend()
    plt.subplot(2, 2, 4)
    plt.plot(steps_epoch, val_epoch_accs, label='Val/Epoch Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Val/Epoch Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'full_learning_curves.png'))
    plt.close()

def plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, save_dir):
    """
        绘制准确率、F1 score、召回率曲线
    """
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_accs, label='Train Accuracy')
    plt.plot(epochs, val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.plot(epochs, train_f1s, label='Train F1')
    plt.plot(epochs, val_f1s, label='Val F1')
    plt.xlabel('Epoch')
    plt.ylabel('Weighted F1-score')
    plt.title('F1 Score Curve')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.plot(epochs, train_recalls, label='Train Recall')
    plt.plot(epochs, val_recalls, label='Val Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.title('Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'metrics_curves.png'))
    plt.close()

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
    results = {
        'test_accuracy': test_acc,
        'auc': auc,
        'weighted_f1': weighted_f1,
        'classification_report': report,
        'confusion_matrix': cm.tolist()
    }
    with open(os.path.join(config.training.log_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    return test_acc, auc, cm, report

def main():
    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = Config()
    # 配置文件配置BackboneConfig
    # config.backbone.pretrained = True
    # config.backbone.pretrained_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    
    # 设置随机种子
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # 创建模型
    model = build_mosa_net(config)
    model = model.to(config.device)
    
    # 打印模型参数数量
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # 创建优化器、损失函数和调度器
    optimizer = create_optimizer(model, config)
    criterion = create_loss_fn(config)
    scheduler = create_scheduler(optimizer, config, train_loader)
    
    # 加载检查点（如果存在）
    start_epoch = 0
    best_acc = 0
    if config.training.resume:
        start_epoch, best_acc = load_checkpoint(
            config.training.resume, model, optimizer, scheduler
        )
        print(f"Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%")
    
    # 训练循环
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_recalls, val_recalls = [], []
    train_aucs, val_aucs = [], []
    train_batch_losses = []
    train_f1s, val_f1s = [], []

    print(f"Batch size: {config.training.batch_size}")
    print(f"Total epochs: {config.training.num_epochs}")

    for epoch in range(start_epoch, config.training.num_epochs):
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_auc, batch_losses, train_targets, train_preds = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, config
        )
        # 验证
        val_loss, val_acc, val_recall, val_auc, val_targets, val_preds = validate(model, val_loader, criterion, config)
        # 更新学习率（对于plateau调度器）
        if scheduler and config.training.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        # 保存最佳模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        if (epoch + 1) % config.training.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'config': config.__dict__
            }, is_best, config.training.checkpoint_dir)
        print(f'Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.2f}, Train AUC: {train_auc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.2f}, Val AUC: {val_auc:.4f}, Best Acc: {best_acc:.2f}%')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_aucs.append(train_auc)
        val_aucs.append(val_auc)
        train_batch_losses.extend(batch_losses)
        # 统计F1-score
        train_report = classification_report(train_targets, train_preds, target_names=['CN', 'AD'], output_dict=True)
        val_report = classification_report(val_targets, val_preds, target_names=['CN', 'AD'], output_dict=True)
        train_f1s.append(train_report['weighted avg']['f1-score'])
        val_f1s.append(val_report['weighted avg']['f1-score'])

    # 绘制详细学习曲线
    plot_full_learning_curves(
        train_batch_losses, train_losses, val_losses, val_accs, config.training.log_dir
    )
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, config.training.log_dir)
    # 绘制指标曲线
    plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, config.training.log_dir)
    print("\n训练完成，开始在测试集上评估最佳模型...")
    best_model_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        load_checkpoint(best_model_path, model)
    else:
        print("未找到最佳模型，使用最终模型进行评估")
    test_acc, auc, cm, report = test_model(model, test_loader, config)
    print(f"\n最终测试结果:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    end_time = datetime.datetime.now()
    print(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"训练总耗时: {str(duration)}")

if __name__ == '__main__':
    main()