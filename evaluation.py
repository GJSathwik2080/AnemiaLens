"""
evaluation.py — Plots and metric tables.
"""

import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import confusion_matrix
from config import Config


def plot_training(hist1, hist2, modality: str):
    Config.ensure_dirs()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    loss   = hist1.history["loss"]     + hist2.history["loss"]
    vloss  = hist1.history["val_loss"] + hist2.history["val_loss"]
    split  = len(hist1.history["loss"])

    ax = axes[0]
    ax.plot(loss,  label="Train")
    ax.plot(vloss, label="Val")
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — Loss", xlabel="Epoch", ylabel="Loss")
    ax.legend(); ax.grid(alpha=.3)

    key = "mae" if Config.TASK_TYPE == "regression" else "accuracy"
    tm = hist1.history[key]       + hist2.history[key]
    vm = hist1.history[f"val_{key}"] + hist2.history[f"val_{key}"]

    ax = axes[1]
    ax.plot(tm, label="Train")
    ax.plot(vm, label="Val")
    ax.axvline(split, ls="--", c="green", label="Fine-tune start")
    ax.set(title=f"{modality} — {key.upper()}", xlabel="Epoch", ylabel=key)
    ax.legend(); ax.grid(alpha=.3)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR,
                             f"{modality}_curves.png"), dpi=150)
    plt.close()


def plot_scatter(y_true, y_pred, title="Hb Regression"):
    Config.ensure_dirs()
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=.5)
    lo = min(y_true.min(), y_pred.min()) - 1
    hi = max(y_true.max(), y_pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.axhline(Config.HB_THRESHOLD, ls=":", c="orange", alpha=.6)
    ax.axvline(Config.HB_THRESHOLD, ls=":", c="orange", alpha=.6)
    ax.set(xlabel="True Hb (g/dL)", ylabel="Predicted Hb (g/dL)", title=title)
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR,
                             f"{title.replace(' ','_')}.png"), dpi=150)
    plt.close()


def plot_confusion(y_true, y_pred_class, title="Confusion Matrix"):
    Config.ensure_dirs()
    cm = confusion_matrix(y_true, y_pred_class)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Anemic"],
                yticklabels=["Healthy","Anemic"], ax=ax)
    ax.set(xlabel="Predicted", ylabel="Actual", title=title)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.RESULTS_DIR,
                             f"{title.replace(' ','_')}.png"), dpi=150)
    plt.close()
