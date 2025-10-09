"""MOSA Net training script

F1 logging policy:
    - Batch-level F1 (Train/F1_batch, Val/F1_batch): macro average (robust to temporary class absence in small batches).
    - Epoch/Test-level F1 (Train/F1, Val/F1, Test/F1): weighted average (reflects class imbalance in overall performance).
"""
import matplotlib
matplotlib.use('Agg')

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
from sklearn.metrics import recall_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, precision_score
from metrics_utils import (
    compute_auc, compute_precision, compute_specificity, compute_f1,
    compute_recall, compute_confusion_matrix, plot_confusion_matrix_figure,
    plot_roc_figure, compute_per_class_metrics, plot_per_class_bars,
    compute_macro_weighted_summary, plot_class_support_bar
)
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# 常量，避免重复字面量
ACC_LABEL = 'Accuracy (%)'
WEIGHTED_AVG_KEY = 'weighted avg'

from config import Config
from data_loader import get_data_loaders
from model_builder import build_mosa_net, count_parameters
from utils import AverageMeter, accuracy, save_checkpoint, load_checkpoint, FocalLoss, plot_confusion_matrix

def log_epoch_metrics(writer, epoch: int, prefix: str, metrics: dict):
    """统一写入 epoch 级指标"""
    if writer is None:
        return
    for k, v in metrics.items():
        writer.add_scalar(f'{prefix}/{k}', v, epoch)

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

def train_epoch(model, train_loader, optimizer, criterion, scheduler, epoch, config, writer=None):
    """训练一个epoch"""
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []
    grad_norm_accumulate = []
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
        # 计算梯度范数
        if writer is not None:
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm_sq += param_norm.item() ** 2
            grad_norm = total_norm_sq ** 0.5
            grad_norm_accumulate.append(grad_norm)
        else:
            grad_norm = None
        optimizer.step()
        if scheduler and config.training.lr_scheduler == "cosine":
            scheduler.step()
        acc1 = accuracy(output, target, topk=(1,))[0]
        losses.update(loss.item(), data.size(0))
        top1.update(acc1.item(), data.size(0))
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)
        batch_targets = target.cpu().numpy()
        batch_preds = preds.cpu().numpy()
        recall = compute_recall(batch_targets, batch_preds)
        recalls.update(recall, data.size(0))
        all_targets.extend(batch_targets)
        all_preds.extend(batch_preds)
        all_probs.extend(probs.detach().cpu().numpy())
        # TensorBoard记录每batch
        if writer is not None:
            f1_batch = compute_f1(batch_targets, batch_preds, average='macro')
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Loss_batch', loss.item(), global_step)
            writer.add_scalar('Train/Accuracy_batch', acc1.item(), global_step)
            writer.add_scalar('Train/Recall_batch', recall, global_step)
            writer.add_scalar('Train/F1_batch', f1_batch, global_step)
            if grad_norm is not None:
                writer.add_scalar('Train/GradNorm_batch', grad_norm, global_step)
    # AUC 使用概率
    auc_val = compute_auc(all_targets, all_probs)
    # f1_epoch 计算在主循环中进行，这里不再需要局部变量，保持返回签名兼容
    if writer is not None and grad_norm_accumulate:
        writer.add_scalar('Train/GradNorm_epoch_mean', float(np.mean(grad_norm_accumulate)), epoch)
    # 训练阶段 Precision / Specificity（epoch级）
    train_precision = compute_precision(all_targets, all_preds)
    train_specificity = compute_specificity(all_targets, all_preds)
    if writer is not None:
        writer.add_scalar('Train/Precision', train_precision, epoch)
        writer.add_scalar('Train/Specificity', train_specificity, epoch)
    return losses.avg, top1.avg, recalls.avg, auc_val, batch_losses, all_targets, all_preds, all_probs

def validate(model, val_loader, criterion, config, epoch=None, writer=None):
    """验证模型"""
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    recalls = AverageMeter()
    all_targets = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for batch_idx, (data, target, _) in enumerate(pbar):
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            loss = criterion(output, target)
            acc1 = accuracy(output, target, topk=(1,))[0]
            losses.update(loss.item(), data.size(0))
            top1.update(acc1.item(), data.size(0))
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            batch_targets = target.cpu().numpy()
            batch_preds = preds.cpu().numpy()
            recall = compute_recall(batch_targets, batch_preds)
            recalls.update(recall, data.size(0))
            all_targets.extend(batch_targets)
            all_preds.extend(batch_preds)
            all_probs.extend(probs.detach().cpu().numpy())
            # F1-score (macro)
            f1 = compute_f1(batch_targets, batch_preds, average='macro')
            # TensorBoard记录每batch
            if writer is not None and epoch is not None:
                global_step = epoch * len(val_loader) + batch_idx
                writer.add_scalar('Val/Loss_batch', loss.item(), global_step)
                writer.add_scalar('Val/Accuracy_batch', acc1.item(), global_step)
                writer.add_scalar('Val/Recall_batch', recall, global_step)
                writer.add_scalar('Val/F1_batch', f1, global_step)
    auc_val = compute_auc(all_targets, all_probs)
    # epoch 级 F1 在主训练循环进行（weighted）；此处无需再计算
    if writer is not None and epoch is not None:
        writer.add_scalar('Val/Loss', losses.avg, epoch)
        writer.add_scalar('Val/Accuracy', top1.avg, epoch)
        writer.add_scalar('Val/Recall', recalls.avg, epoch)
        writer.add_scalar('Val/AUC', auc_val, epoch)
    # 验证阶段 Precision / Specificity
    val_precision = compute_precision(all_targets, all_preds)
    val_specificity = compute_specificity(all_targets, all_preds)
    if writer is not None and epoch is not None:
        writer.add_scalar('Val/Precision', val_precision, epoch)
        writer.add_scalar('Val/Specificity', val_specificity, epoch)
    return losses.avg, top1.avg, recalls.avg, auc_val, all_targets, all_preds, all_probs

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
    plt.ylabel(ACC_LABEL)
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
    plt.ylabel(ACC_LABEL)
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
    plt.ylabel(ACC_LABEL)
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

def test_model(model, test_loader, config, class_names=['CN', 'AD'], save_dir=None):
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
    out_dir = save_dir or config.training.log_dir
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(out_dir, 'confusion_matrix.png'))
    weighted_f1 = report[WEIGHTED_AVG_KEY]['f1-score']
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
    with open(os.path.join(out_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    # Precision / Specificity / ROC
    try:
        test_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    except Exception:
        test_precision = 0.0
    try:
        specs = []
        for i in range(cm.shape[0]):
            TP = cm[i, i]
            FP = cm[:, i].sum() - TP
            FN = cm[i, :].sum() - TP
            TN = cm.sum() - TP - FP - FN
            spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
            specs.append(spec)
        test_specificity = float(np.mean(specs)) if specs else 0.0
    except Exception:
        test_specificity = 0.0
    # 写入 TensorBoard（若存在全局 writer，在 main 中执行） -> 返回扩展指标
    return test_acc, auc, cm, report, test_precision, test_specificity, all_targets, all_probs

def main():
    # 记录训练开始时间
    start_time = datetime.datetime.now()
    print(f"训练开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 加载配置
    config = Config()
    # 可按需在此处覆写 config.backbone 相关参数
    
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
    
    # TensorBoard记录: 独立 run 目录
    run_name = f"mosa_net_bs{config.data.batch_size}_lr{config.training.learning_rate}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    log_dir = os.path.join(config.training.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    try:
        writer.add_text('Config', json.dumps(config.__dict__, indent=2))
    except Exception:
        pass
    
    # 训练循环
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    train_recalls, val_recalls = [], []
    train_batch_losses = []
    train_f1s, val_f1s = [], []
    best_state = {'epoch': -1, 'val_acc': -1, 'val_auc': 0, 'val_f1': 0, 'val_recall': 0}

    print(f"Batch size: {config.data.batch_size}")
    print(f"Total epochs: {config.training.num_epochs}")

    for epoch in range(start_epoch, config.training.num_epochs):
        # 训练一个epoch
        train_loss, train_acc, train_recall, train_auc, batch_losses, train_targets, train_preds, _ = train_epoch(
            model, train_loader, optimizer, criterion, scheduler, epoch, config, writer
        )
        # 验证
        val_loss, val_acc, val_recall, val_auc, val_targets, val_preds, val_probs = validate(
            model, val_loader, criterion, config, epoch, writer
        )

        from sklearn.metrics import f1_score
        train_f1 = f1_score(train_targets, train_preds, average='weighted', zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average='weighted', zero_division=0)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_f1s.append(train_f1)
        val_f1s.append(val_f1)
        train_recalls.append(train_recall)
        val_recalls.append(val_recall)
        train_batch_losses.extend(batch_losses)

        # 每 epoch 写入统一命名
        log_epoch_metrics(writer, epoch, 'Train', {
            'Loss': train_loss,
            'Accuracy': train_acc,
            'F1': train_f1,
            'Recall': train_recall,
            'AUC': train_auc,
        })
        log_epoch_metrics(writer, epoch, 'Val', {
            'Loss': val_loss,
            'Accuracy': val_acc,
            'F1': val_f1,
            'Recall': val_recall,
            'AUC': val_auc,
        })
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        if (epoch + 1) % 5 == 0:
            writer.flush()

        # batch 级写入已在 train_epoch 内完成，无需重复
        # 更新学习率（对于plateau调度器）
        if scheduler and config.training.lr_scheduler == "plateau":
            scheduler.step(val_acc)
        # 保存最佳模型
        is_best = val_acc > best_state['val_acc']
        if is_best:
            best_state.update({'epoch': epoch, 'val_acc': val_acc, 'val_auc': val_auc, 'val_f1': val_f1, 'val_recall': val_recall})
            best_acc = val_acc
            # 写入最佳混淆矩阵与 ROC (+ 可选 per-class 条形图/JSON)
            try:
                cm_best = confusion_matrix(val_targets, val_preds)
                fig_cm, ax = plt.subplots(figsize=(4,4))
                im = ax.imshow(cm_best, cmap='Blues')
                ax.set_title('Best Val Confusion Matrix')
                ax.set_xlabel('Pred')
                ax.set_ylabel('True')
                for i in range(len(cm_best)):
                    for j in range(len(cm_best)):
                        ax.text(j, i, cm_best[i, j], ha='center', va='center', color='black')
                fig_cm.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                writer.add_figure('Val/ConfusionMatrix_best', fig_cm, epoch)
                plt.close(fig_cm)
                if getattr(getattr(config, 'logging', None), 'export_per_class', True):
                    # per-class metrics bar
                    try:
                        per_cls = compute_per_class_metrics(val_targets, val_preds)
                        fig_bar = plot_per_class_bars(per_cls, title='Best Val Per-class Metrics')
                        if fig_bar:
                            writer.add_figure('Val/PerClassMetrics_best', fig_bar, epoch)
                            plt.close(fig_bar)
                        fig_sup = plot_class_support_bar(per_cls, title='Best Val Class Support')
                        if fig_sup:
                            writer.add_figure('Val/ClassSupport_best', fig_sup, epoch)
                            plt.close(fig_sup)
                        summary = compute_macro_weighted_summary(val_targets, val_preds)
                        # JSON 保存
                        try:
                            import json as _json
                            per_cls_out = {**per_cls, 'epoch': epoch, 'type': 'best_val', 'summary': summary}
                            with open(os.path.join(log_dir, 'best_val_per_class_metrics.json'), 'w') as fpc:
                                _json.dump(per_cls_out, fpc, indent=2)
                        except Exception:
                            pass
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                if len(set(val_targets)) == 2:
                    y_true = np.array(val_targets)
                    y_prob = np.array(val_probs)
                    fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
                    best_roc_auc = roc_auc_score(y_true, y_prob[:,1])
                    fig_roc, axr = plt.subplots(figsize=(4,4))
                    axr.plot(fpr, tpr, label=f'ROC AUC={best_roc_auc:.4f}')
                    axr.plot([0,1],[0,1],'--', color='gray')
                    axr.set_xlabel('FPR')
                    axr.set_ylabel('TPR')
                    axr.set_title('Best Val ROC')
                    axr.legend(loc='lower right')
                    writer.add_figure('Val/ROC_best', fig_roc, epoch)
                    plt.close(fig_roc)
            except Exception:
                pass
        if (epoch + 1) % config.training.save_freq == 0 or is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'config': config.__dict__
            }, is_best, config.training.checkpoint_dir)
        print(f'Epoch {epoch+1}: '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Recall: {train_recall:.4f} | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Recall: {val_recall:.4f}')
    # （以上指标已经在前面 append，不再重复，移除重复代码）

    # 绘制详细学习曲线
    plot_full_learning_curves(
        train_batch_losses, train_losses, val_losses, val_accs, log_dir
    )
    # 绘制学习曲线
    plot_learning_curves(train_losses, val_losses, train_accs, val_accs, log_dir)
    # 绘制指标曲线
    plot_metrics_curves(train_accs, val_accs, train_f1s, val_f1s, train_recalls, val_recalls, log_dir)
    print("\n训练完成，开始在测试集上评估最佳模型...")
    best_model_path = os.path.join(config.training.checkpoint_dir, 'model_best.pth.tar')
    if os.path.exists(best_model_path):
        print(f"加载最佳模型: {best_model_path}")
        load_checkpoint(best_model_path, model)
    else:
        print("未找到最佳模型，使用最终模型进行评估")
    test_acc, test_auc, cm_test, _, test_precision, test_specificity, test_targets, test_probs = test_model(model, test_loader, config, save_dir=log_dir)
    # Test ROC & ConfusionMatrix & per-class metrics to TensorBoard
    try:
        if len(set(test_targets)) == 2:
            y_true = np.array(test_targets)
            y_prob = np.array(test_probs)
            fpr, tpr, _ = roc_curve(y_true, y_prob[:,1])
            roc_auc = roc_auc_score(y_true, y_prob[:,1])
            fig_roc, axr = plt.subplots(figsize=(4,4))
            axr.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.4f}')
            axr.plot([0,1],[0,1],'--', color='gray')
            axr.set_xlabel('FPR'); axr.set_ylabel('TPR'); axr.set_title('Test ROC'); axr.legend(loc='lower right')
            writer.add_figure('Test/ROC', fig_roc, 0)
            plt.close(fig_roc)
        # Confusion matrix figure
        fig_cm, axc = plt.subplots(figsize=(4,4))
        im = axc.imshow(cm_test, cmap='Blues')
        axc.set_title('Test Confusion Matrix')
        axc.set_xlabel('Pred'); axc.set_ylabel('True')
        for i in range(len(cm_test)):
            for j in range(len(cm_test)):
                axc.text(j, i, cm_test[i, j], ha='center', va='center', color='black')
        fig_cm.colorbar(im, ax=axc, fraction=0.046, pad=0.04)
        writer.add_figure('Test/ConfusionMatrix', fig_cm, 0)
        plt.close(fig_cm)
        # Scalar test metrics + per-class bars
        from sklearn.metrics import f1_score as _f1
        test_f1 = _f1(test_targets, np.argmax(np.array(test_probs), axis=1), average='weighted', zero_division=0)
        writer.add_scalar('Test/Accuracy', test_acc, 0)
        writer.add_scalar('Test/AUC', test_auc, 0)
        writer.add_scalar('Test/F1', test_f1, 0)
        writer.add_scalar('Test/Precision', test_precision, 0)
        writer.add_scalar('Test/Specificity', test_specificity, 0)
        if getattr(getattr(config, 'logging', None), 'export_per_class', True):
            try:
                per_cls_test = compute_per_class_metrics(test_targets, np.argmax(np.array(test_probs), axis=1))
                fig_bar_test = plot_per_class_bars(per_cls_test, title='Test Per-class Metrics')
                if fig_bar_test:
                    writer.add_figure('Test/PerClassMetrics', fig_bar_test, 0)
                    plt.close(fig_bar_test)
                fig_sup_test = plot_class_support_bar(per_cls_test, title='Test Class Support')
                if fig_sup_test:
                    writer.add_figure('Test/ClassSupport', fig_sup_test, 0)
                    plt.close(fig_sup_test)
                summary_test = compute_macro_weighted_summary(test_targets, np.argmax(np.array(test_probs), axis=1))
                # 写出测试 per-class JSON
                try:
                    import json as _json
                    per_cls_out_t = {**per_cls_test, 'type': 'test', 'summary': summary_test}
                    with open(os.path.join(log_dir, 'test_per_class_metrics.json'), 'w') as fpt:
                        _json.dump(per_cls_out_t, fpt, indent=2)
                except Exception:
                    pass
            except Exception:
                pass
    except Exception:
        pass
    print("\n最终测试结果:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {test_auc:.4f}")
    end_time = datetime.datetime.now()
    print(f"训练结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end_time - start_time
    print(f"训练总耗时: {str(duration)}")
    # 写入 hparams
    try:
        hparams = {
            'lr': config.training.learning_rate,
            'batch_size': config.data.batch_size,
            'optimizer': config.training.optimizer,
            'loss_fn': config.training.loss_fn,
            'epochs': config.training.num_epochs,
        }
        metrics = {
            'hparam/best_val_acc': best_state['val_acc'],
            'hparam/best_val_auc': best_state['val_auc'],
            'hparam/best_val_f1': best_state['val_f1'],
            'hparam/best_val_recall': best_state['val_recall'],
        }
        writer.add_hparams(hparams, metrics)
    except Exception:
        pass
    # 统一输出 summary_all.json（best validation + test 聚合）
    try:
        summary_all = {
            'best_val': {
                'epoch': best_state.get('epoch', -1),
                'accuracy': best_state.get('val_acc', 0),
                'auc': best_state.get('val_auc', 0),
                'f1_weighted': best_state.get('val_f1', 0),
                'recall_macro': best_state.get('val_recall', 0)
            },
            'test': {
                'accuracy': test_acc,
                'auc': test_auc,
                'f1_weighted': (locals().get('test_f1') if 'test_f1' in locals() else compute_f1(test_targets, np.argmax(np.array(test_probs), axis=1), average='weighted')),
                'precision_weighted': locals().get('test_precision', None),
                'specificity_macro': locals().get('test_specificity', None)
            },
            'config': {
                'optimizer': config.training.optimizer,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.data.batch_size,
                'epochs': config.training.num_epochs,
                'loss_fn': config.training.loss_fn,
                'backbone': config.backbone.model_type,
                'num_classes': config.backbone.num_classes
            }
        }
        with open(os.path.join(log_dir, 'summary_all.json'), 'w') as sf:
            json.dump(summary_all, sf, indent=2)
    except Exception:
        pass
    writer.close()

if __name__ == '__main__':
    main()