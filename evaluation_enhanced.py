"""
evaluation_enhanced.py — Advanced metrics and evaluation with F1, AUC, precision, recall.

Key improvements:
- Per-modality F1, precision, recall, AUC
- Ensemble performance metrics
- Calibration curves
- ROC curves
- Better visualization
- Comprehensive classification report
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss
)
from config import Config


class EvaluationMetrics:
    """Comprehensive evaluation metrics for binary classification."""
    
    def __init__(self, modality: str):
        self.modality = modality
        self.metrics = {}
        
    def calculate(self, y_true, y_pred_prob, threshold=0.5):
        """
        Calculate comprehensive metrics.
        
        Args:
            y_true: ground truth labels (0/1 or continuous 0-1)
            y_pred_prob: predicted probabilities [0, 1]
            threshold: classification threshold
        """
        # Convert to binary if needed
        y_binary = (y_true > threshold).astype(int)
        y_pred_class = (y_pred_prob > threshold).astype(int)
        
        # Basic metrics
        self.metrics['accuracy'] = np.mean(y_binary == y_pred_class)
        self.metrics['f1'] = f1_score(y_binary, y_pred_class, zero_division=0)
        self.metrics['precision'] = precision_score(y_binary, y_pred_class, zero_division=0)
        self.metrics['recall'] = recall_score(y_binary, y_pred_class, zero_division=0)
        
        # ROC-AUC
        try:
            self.metrics['auc'] = roc_auc_score(y_binary, y_pred_prob)
        except:
            self.metrics['auc'] = 0.5
        
        # Log loss
        self.metrics['logloss'] = log_loss(y_binary, y_pred_prob)
        
        # Sensitivity and specificity
        tn, fp, fn, tp = confusion_matrix(y_binary, y_pred_class).ravel()
        self.metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return self.metrics
    
    def get_report(self):
        """Return formatted metrics report."""
        report = f"\n{'='*60}\n"
        report += f"METRICS — {self.modality.upper()}\n"
        report += f"{'='*60}\n"
        for key, val in self.metrics.items():
            report += f"  {key:15s}: {val:.4f}\n"
        report += f"{'='*60}\n"
        return report
    
    def print_report(self):
        print(self.get_report())


def plot_training_curves_enhanced(hist1, hist2, modality: str):
    """Plot training curves with multiple metrics."""
    Config.ensure_dirs()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    loss = hist1.history["loss"] + hist2.history["loss"]
    vloss = hist1.history["val_loss"] + hist2.history["val_loss"]
    split = len(hist1.history["loss"])
    
    # Loss
    ax = axes[0, 0]
    ax.plot(loss, label="Train Loss", linewidth=2)
    ax.plot(vloss, label="Val Loss", linewidth=2)
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — Loss", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Accuracy/MAE
    key = "mae" if Config.TASK_TYPE == "regression" else "accuracy"
    tm = hist1.history[key] + hist2.history[key]
    vm = hist1.history[f"val_{key}"] + hist2.history[f"val_{key}"]
    
    ax = axes[0, 1]
    ax.plot(tm, label=f"Train {key.upper()}", linewidth=2)
    ax.plot(vm, label=f"Val {key.upper()}", linewidth=2)
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — {key.upper()}", xlabel="Epoch", ylabel=key)
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Overfitting analysis
    ax = axes[1, 0]
    overfitting = np.array(vloss) - np.array(loss)
    ax.plot(overfitting, label="Val - Train Loss", linewidth=2, color='orange')
    ax.axhline(0, ls="--", c="gray", alpha=0.5)
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — Overfitting Analysis", xlabel="Epoch", ylabel="Difference")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Learning stability
    ax = axes[1, 1]
    train_smoothed = np.convolve(loss, np.ones(5)/5, mode='valid')
    val_smoothed = np.convolve(vloss, np.ones(5)/5, mode='valid')
    ax.plot(train_smoothed, label="Train (smoothed)", linewidth=2)
    ax.plot(val_smoothed, label="Val (smoothed)", linewidth=2)
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — Learning Stability", xlabel="Epoch", ylabel="Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{modality}_training_curves_enhanced.png"), dpi=150)
    plt.close()


def plot_roc_curve(y_true, y_pred_prob, modality: str, title: str = "ROC Curve"):
    """Plot ROC curve."""
    Config.ensure_dirs()
    
    y_binary = (y_true > 0.5).astype(int)
    fpr, tpr, _ = roc_curve(y_binary, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='False Positive Rate',
           ylabel='True Positive Rate', title=f"{title} — {modality.upper()}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{modality}_roc_curve.png"), dpi=150)
    plt.close()
    
    return roc_auc


def plot_precision_recall_curve(y_true, y_pred_prob, modality: str):
    """Plot precision-recall curve."""
    Config.ensure_dirs()
    
    y_binary = (y_true > 0.5).astype(int)
    precision, recall, _ = precision_recall_curve(y_binary, y_pred_prob)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set(xlim=[0.0, 1.0], ylim=[0.0, 1.05], xlabel='Recall',
           ylabel='Precision', title=f"Precision-Recall — {modality.upper()}")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{modality}_pr_curve.png"), dpi=150)
    plt.close()
    
    return pr_auc


def plot_confusion_matrix_enhanced(y_true, y_pred_prob, modality: str, threshold=0.5):
    """Plot confusion matrix with better formatting."""
    Config.ensure_dirs()
    
    y_binary = (y_true > threshold).astype(int)
    y_pred_class = (y_pred_prob > threshold).astype(int)
    
    cm = confusion_matrix(y_binary, y_pred_class)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Anemic'],
                yticklabels=['Healthy', 'Anemic'],
                ax=ax, cbar_kws={'label': 'Count'})
    
    ax.set(xlabel='Predicted Label', ylabel='True Label',
           title=f"Confusion Matrix — {modality.upper()}")
    
    # Add percentages
    for i in range(2):
        for j in range(2):
            total = cm.sum()
            percentage = cm[i, j] / total * 100 if total > 0 else 0
            ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)', 
                   ha='center', va='center', color='gray', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{modality}_confusion_matrix_enhanced.png"), dpi=150)
    plt.close()


def plot_calibration_curve(y_true, y_pred_prob, modality: str, n_bins=10):
    """Plot calibration curve to check if predicted probabilities match actual accuracy."""
    Config.ensure_dirs()
    
    y_binary = (y_true > 0.5).astype(int)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Perfect calibration line
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    
    # Binned calibration
    bin_sums = np.bincount(np.digitize(y_pred_prob, np.linspace(0, 1, n_bins)) - 1, minlength=n_bins)
    bin_true = np.bincount(np.digitize(y_pred_prob, np.linspace(0, 1, n_bins)) - 1, 
                           weights=y_binary, minlength=n_bins)
    bin_prob = np.bincount(np.digitize(y_pred_prob, np.linspace(0, 1, n_bins)) - 1,
                           weights=y_pred_prob, minlength=n_bins)
    
    # Compute calibration points
    nonzero = bin_sums != 0
    prob_true = bin_true[nonzero] / bin_sums[nonzero]
    prob_pred = bin_prob[nonzero] / bin_sums[nonzero]
    
    ax.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='Model')
    ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
           xlabel='Mean predicted probability', ylabel='Fraction of positives',
           title=f"Calibration Curve — {modality.upper()}")
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR, f"{modality}_calibration_curve.png"), dpi=150)
    plt.close()


def generate_evaluation_report(y_true_dict, y_pred_dict, modality_list=['conjunctiva', 'nail', 'palm']):
    """Generate comprehensive evaluation report for all modalities."""
    
    report = "\n" + "="*80 + "\n"
    report += "COMPREHENSIVE EVALUATION REPORT\n"
    report += "="*80 + "\n\n"
    
    all_metrics = {}
    
    for modality in modality_list:
        if modality not in y_true_dict:
            continue
        
        y_true = y_true_dict[modality]
        y_pred = y_pred_dict[modality]
        
        metrics = EvaluationMetrics(modality)
        metrics.calculate(y_true, y_pred)
        all_metrics[modality] = metrics.metrics
        
        report += metrics.get_report()
    
    # Ensemble metrics if all modalities present
    if len(all_metrics) == 3:
        report += "\n" + "="*80 + "\n"
        report += "ENSEMBLE AVERAGE METRICS\n"
        report += "="*80 + "\n"
        
        avg_metrics = {}
        for key in all_metrics['conjunctiva'].keys():
            values = [all_metrics[m][key] for m in modality_list]
            avg_metrics[key] = np.mean(values)
        
        for key, val in avg_metrics.items():
            report += f"  {key:15s}: {val:.4f}\n"
    
    return report
