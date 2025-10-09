import torch
from config import Config
from data_loader import get_data_loaders
from model_builder import build_backbone, build_mosa_net
from utils import load_checkpoint, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from metrics_utils import compute_auc, compute_per_class_metrics, plot_per_class_bars, compute_macro_weighted_summary, plot_class_support_bar, compute_f1, compute_precision, compute_specificity
import numpy as np
import os
import datetime
import subprocess
from tqdm import tqdm

def test_model(model, test_loader, config, class_names=['CN', 'AD']):
    """测试模型性能 (增强版)"""
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Testing')
        for data, target, _ in pbar:
            data, target = data.to(config.device), target.to(config.device)
            output = model(data)
            probs = torch.softmax(output, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    # 指标
    auc = compute_auc(all_targets, all_probs)
    cm = confusion_matrix(all_targets, all_preds)
    report = classification_report(all_targets, all_preds, target_names=class_names, output_dict=True)
    test_acc = np.mean(np.array(all_preds) == np.array(all_targets)) * 100
    weighted_f1 = compute_f1(all_targets, all_preds, average='weighted')
    test_precision = compute_precision(all_targets, all_preds)
    test_specificity = compute_specificity(all_targets, all_preds)
    # 可视化
    plot_confusion_matrix(cm, class_names, save_path=os.path.join(config.training.log_dir, 'confusion_matrix.png'))
    # ROC 图
    try:
        import matplotlib.pyplot as plt
        if len(set(all_targets)) == 2:
            y_true = np.array(all_targets)
            y_prob = np.array(all_probs)[:,1]
            from sklearn.metrics import roc_curve, auc as _auc
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_val = _auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(fpr, tpr, label=f'AUC={roc_val:.4f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Test ROC')
            ax.legend(loc='lower right')
            fig.savefig(os.path.join(config.training.log_dir, 'test_roc.png'))
            plt.close(fig)
        else:
            from sklearn.preprocessing import label_binarize
            from sklearn.metrics import roc_curve, auc as _auc
            y_true = np.array(all_targets)
            classes = sorted(set(all_targets))
            Y = label_binarize(y_true, classes=classes)
            probs_arr = np.array(all_probs)
            fpr, tpr, _ = roc_curve(Y.ravel(), probs_arr[:, :len(classes)].ravel())
            roc_val = _auc(fpr, tpr)
            fig, ax = plt.subplots(figsize=(4,4))
            ax.plot(fpr, tpr, label=f'Micro AUC={roc_val:.4f}')
            ax.plot([0,1],[0,1],'--', color='gray')
            ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('Test ROC (micro)')
            ax.legend(loc='lower right')
            fig.savefig(os.path.join(config.training.log_dir, 'test_roc.png'))
            plt.close(fig)
    except Exception:
        pass
    # 控制台输出
    print(f"\nWeighted F1-score: {weighted_f1:.4f}")
    print("\n" + "="*50)
    print("测试集性能评估")
    print("="*50)
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")
    print(f"Weighted F1-score: {weighted_f1:.4f}")
    print(f"Precision (weighted): {test_precision:.4f}")
    print(f"Specificity (macro): {test_specificity:.4f}")
    from sklearn.metrics import classification_report as _cr
    print("\n分类报告:")
    print(_cr(all_targets, all_preds, target_names=class_names))
    print("\n混淆矩阵:")
    print(cm)
    per_class_json = None
    if getattr(getattr(config, 'logging', None), 'export_per_class', True):
        try:
            per_cls = compute_per_class_metrics(all_targets, all_preds)
            summary = compute_macro_weighted_summary(all_targets, all_preds)
            per_class_json = {**per_cls, 'summary': summary, 'type': 'test'}
            fig_bar = plot_per_class_bars(per_cls, title='Test Per-class Metrics')
            if fig_bar:
                fig_bar.savefig(os.path.join(config.training.log_dir, 'test_per_class_metrics.png'))
            fig_sup = plot_class_support_bar(per_cls, title='Test Class Support')
            if fig_sup:
                fig_sup.savefig(os.path.join(config.training.log_dir, 'test_class_support.png'))
        except Exception:
            pass
    # summary_all.json 输出
    try:
        import json as _json
        run_end = datetime.datetime.now(datetime.timezone.utc)
        run_start = getattr(config, '_run_start_time', run_end)
        duration = (run_end - run_start).total_seconds()
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            git_commit = None
        summary_all = {
            'best_val': None,
            'test': {
                'accuracy': test_acc,
                'auc': auc,
                'f1_weighted': weighted_f1,
                'precision_weighted': test_precision,
                'specificity_macro': test_specificity
            },
            'config': {
                'optimizer': config.training.optimizer,
                'learning_rate': config.training.learning_rate,
                'batch_size': config.data.batch_size,
                'loss_fn': config.training.loss_fn,
                'backbone': config.backbone.model_type,
                'num_classes': config.backbone.num_classes
            },
            'run_meta': {
                'run_start_time_utc': run_start.isoformat(),
                'run_end_time_utc': run_end.isoformat(),
                'duration_seconds': duration,
                'git_commit': git_commit
            }
        }
        with open(os.path.join(config.training.log_dir, 'summary_all.json'), 'w') as sf:
            _json.dump(summary_all, sf, indent=2)
        if per_class_json is not None:
            with open(os.path.join(config.training.log_dir, 'test_per_class_metrics.json'), 'w') as pf:
                _json.dump(per_class_json, pf, indent=2)
    except Exception:
        pass
    return test_acc, auc, cm, report

def main():
    # 加载配置
    config = Config()
    config.backbone.pretrained = True
    config.backbone.pretrained_path = "/path/to/resnet_pretrained_weights.pth"  # 替换为ResNet预训练权重路径
    config.training.resume = "/path/to/mosa_simple_trained_weights.pth"  # 替换为MosaSimple训练权重路径

    # 创建数据加载器
    _, _, test_loader = get_data_loaders(config)

    # 构建模型
    model = build_mosa_net(config)
    model = model.to(config.device)

    # 加载MosaSimple训练权重
    if config.training.resume:
        print(f"加载MosaSimple训练权重: {config.training.resume}")
        load_checkpoint(config.training.resume, model)

    # 测试模型
    print("开始测试模型...")
    test_acc, auc, _, _ = test_model(model, test_loader, config)

    print("\n测试完成，结果如下:")
    print(f"测试准确率: {test_acc:.2f}%")
    print(f"AUC: {auc:.4f}")

if __name__ == "__main__":
    main()