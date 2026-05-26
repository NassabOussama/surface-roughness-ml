from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from config import CLASSES


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, grit_idx, labels in loader:
        images, grit_idx, labels = images.to(device), grit_idx.to(device), labels.to(device)
        logits = model(images, grit_idx)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Per-class F1 (macro)
    f1_scores = []
    conf_matrix = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for i in range(len(CLASSES)):
        tp = ((all_preds == i) & (all_labels == i)).sum()
        fp = ((all_preds == i) & (all_labels != i)).sum()
        fn = ((all_preds != i) & (all_labels == i)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_scores.append(2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0)
        for j in range(len(CLASSES)):
            conf_matrix[i, j] = ((all_labels == i) & (all_preds == j)).sum()

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
        "f1": float(np.mean(f1_scores)),
        "per_class_f1": {CLASSES[i]: f1_scores[i] for i in range(len(CLASSES))},
        "confusion_matrix": conf_matrix,
    }


def plot_training_history(history: dict, save_path: Optional[str] = None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train")
    axes[1].plot(history["val_acc"], label="Val")
    axes[1].set_title("Accuracy"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(history["lr"], color="orange")
    axes[2].set_title("Learning Rate"); axes[2].set_xlabel("Epoch")
    axes[2].set_yscale("log"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(matrix: np.ndarray, save_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
