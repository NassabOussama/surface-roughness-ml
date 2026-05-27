"""
Training entry point.

Usage:
    python train.py \
        --images_path /path/to/Images \
        --labels_path /path/to/Labelisation_CSV.csv \
        --model vit_grit \
        --output_dir ./outputs
"""
import argparse
import csv
import os
import statistics
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import mlflow
import mlflow.pytorch

from config import Config, CLASSES
from data.dataset import load_labels, RoughnessDataset, group_split, group_kfold_split
from data.transforms import get_transforms
from models.vit_grit import ViTGRiT
from models.film_resnet50 import FiLMResNet50
from evaluate import evaluate, plot_training_history, plot_confusion_matrix


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience: int, min_delta: float):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = None
        self.best_state = None
        self.stop = False

    def __call__(self, val_acc: float, model: nn.Module) -> bool:
        if self.best is None or val_acc > self.best + self.min_delta:
            self.best = val_acc
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def restore(self, model: nn.Module):
        if self.best_state:
            model.load_state_dict(self.best_state)


def train_one_epoch(model, loader, optimizer, criterion, device, amp_scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, grit_idx, labels in loader:
        images, grit_idx, labels = images.to(device), grit_idx.to(device), labels.to(device)
        optimizer.zero_grad()

        if amp_scaler:
            with torch.cuda.amp.autocast():
                logits = model(images, grit_idx)
                loss = criterion(logits, labels)
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
        else:
            logits = model(images, grit_idx)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


def build_model(config: Config, pretrained: bool = True) -> nn.Module:
    kwargs = dict(
        num_grit_classes=config.num_grit_classes,
        embedding_dim=config.embedding_dim,
        dropout_rate=config.dropout_rate,
        num_classes=config.num_classes,
        pretrained=pretrained,
    )
    if config.model_name == "vit_grit":
        return ViTGRiT(**kwargs)
    if config.model_name == "film_resnet50":
        return FiLMResNet50(**kwargs)
    raise ValueError(f"Unknown model: {config.model_name}")


def save_checkpoint(model, config, metrics, path):
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_name": config.model_name,
        "num_grit_classes": config.num_grit_classes,
        "embedding_dim": config.embedding_dim,
        "dropout_rate": config.dropout_rate,
        "num_classes": config.num_classes,
        "grit_values": list(config.grit_values),
        "metrics": metrics,
    }, path)


def train_eval_once(config: Config, train_rows, val_rows, test_rows, device,
                    run_name: str, output_subdir: str,
                    nested: bool = False, log_full_model: bool = True,
                    extra_params: dict = None):
    """
    Train one model on one (train, val, test) split and return test metrics.

    Training randomness (init, batch order, augmentation) is reseeded here from
    ``config.random_seed`` so it is identical regardless of how the split was
    produced; the split itself is decided by the caller via ``split_seed``.
    """
    set_seed(config.random_seed)
    os.makedirs(output_subdir, exist_ok=True)

    g2i = config.grit_to_idx
    train_ds = RoughnessDataset(train_rows, config.images_path, g2i, get_transforms(config.input_size, "train"))
    val_ds   = RoughnessDataset(val_rows,   config.images_path, g2i, get_transforms(config.input_size, "val"))
    test_ds  = RoughnessDataset(test_rows,  config.images_path, g2i, get_transforms(config.input_size, "test"))
    print(f"Images — train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_model(config).to(device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=config.scheduler_T0, T_mult=config.scheduler_T_mult,
                                            eta_min=config.learning_rate * 0.01)
    criterion = nn.CrossEntropyLoss()
    early_stop = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    amp_scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}

    with mlflow.start_run(run_name=run_name, nested=nested):
        params = {
            "model": config.model_name,
            "lr": config.learning_rate,
            "batch_size": config.batch_size,
            "embedding_dim": config.embedding_dim,
            "dropout_rate": config.dropout_rate,
            "split_seed": config.split_seed,
            "train_seed": config.random_seed,
        }
        if extra_params:
            params.update(extra_params)
        mlflow.log_params(params)

        best_val_acc = 0.0
        for epoch in range(config.num_epochs):
            tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, amp_scaler)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            lr = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_metrics["loss"])
            history["val_acc"].append(val_metrics["accuracy"])
            history["lr"].append(lr)

            mlflow.log_metrics({
                "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            }, step=epoch)

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d} | train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
                      f"val_loss={val_metrics['loss']:.4f} acc={val_metrics['accuracy']:.3f}")

            if early_stop(val_metrics["accuracy"], model):
                print(f"Early stopping at epoch {epoch + 1}")
                break

        early_stop.restore(model)
        model.to(device)

        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Test F1 (macro): {test_metrics['f1']:.4f}")
        mlflow.log_metrics({"test_acc": test_metrics["accuracy"], "test_f1": test_metrics["f1"],
                            "best_val_acc": best_val_acc})

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_path = os.path.join(output_subdir, f"{config.model_name}_{ts}.pth")
        save_checkpoint(model, config, {"val_acc": best_val_acc, **test_metrics}, ckpt_path)

        plot_training_history(history, os.path.join(output_subdir, "training_history.png"))
        plot_confusion_matrix(test_metrics["confusion_matrix"],
                              os.path.join(output_subdir, "confusion_matrix.png"))
        mlflow.log_artifact(os.path.join(output_subdir, "training_history.png"))
        mlflow.log_artifact(os.path.join(output_subdir, "confusion_matrix.png"))

        # Heavy artefacts (344 MB pickled model + checkpoint copy) only when asked.
        # Skipped for k-fold folds to avoid duplicating many GB across mlruns.
        if log_full_model:
            mlflow.log_artifact(ckpt_path)
            mlflow.pytorch.log_model(model, "model")
        print(f"Checkpoint saved: {ckpt_path}")

    return test_metrics


def _aggregate(metrics_list, label_keys):
    """Print and return mean ± sample-std for a list of metric dicts."""
    out = {}
    for key in label_keys:
        vals = [m[key] for m in metrics_list]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        out[key] = (mean, std)
    return out


def run(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    rows = load_labels(config.labels_path)
    mlflow.set_experiment("surface-roughness-classification")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    if config.kfold and config.kfold > 1:
        folds = group_kfold_split(rows, config)
        print(f"{config.kfold}-fold group-stratified CV (split_seed={config.split_seed}, "
              f"train_seed={config.random_seed})")

        parent_name = f"{config.model_name}_cv{config.kfold}_splitseed{config.split_seed}_seed{config.random_seed}_{ts}"
        fold_metrics = []
        with mlflow.start_run(run_name=parent_name):
            mlflow.log_params({
                "model": config.model_name, "kfold": config.kfold,
                "split_seed": config.split_seed, "train_seed": config.random_seed,
            })
            for i, (tr, va, te) in enumerate(folds):
                print(f"\n===== Fold {i + 1}/{config.kfold} =====")
                sub = os.path.join(config.output_dir, f"fold{i}")
                m = train_eval_once(
                    config, tr, va, te, device,
                    run_name=f"{parent_name}_fold{i}", output_subdir=sub,
                    nested=True, log_full_model=False,
                    extra_params={"kfold": config.kfold, "fold": i},
                )
                fold_metrics.append(m)

            agg = _aggregate(fold_metrics, ["accuracy", "f1"])
            (acc_m, acc_s), (f1_m, f1_s) = agg["accuracy"], agg["f1"]
            mlflow.log_metrics({
                "cv_acc_mean": acc_m, "cv_acc_std": acc_s,
                "cv_f1_mean": f1_m, "cv_f1_std": f1_s,
            })

        # Per-fold CSV summary on disk
        summary_path = os.path.join(config.output_dir, "cv_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["fold", "test_acc", "test_f1"])
            for i, m in enumerate(fold_metrics):
                w.writerow([i, f"{m['accuracy']:.4f}", f"{m['f1']:.4f}"])
            w.writerow(["mean", f"{acc_m:.4f}", f"{f1_m:.4f}"])
            w.writerow(["std", f"{acc_s:.4f}", f"{f1_s:.4f}"])

        print(f"\n===== {config.kfold}-fold CV summary ({config.model_name}) =====")
        print(f"Test accuracy: {acc_m:.4f} ± {acc_s:.4f}")
        print(f"Test F1 (macro): {f1_m:.4f} ± {f1_s:.4f}")
        print(f"Summary written: {summary_path}")
        return fold_metrics

    # Single split (default — unchanged behaviour)
    train_rows, val_rows, test_rows = group_split(rows, config)
    print(f"Split — train: {len(train_rows)}, val: {len(val_rows)}, test: {len(test_rows)} samples "
          f"(split_seed={config.split_seed}, train_seed={config.random_seed})")
    run_name = f"{config.model_name}_splitseed{config.split_seed}_seed{config.random_seed}_{ts}"
    test_metrics = train_eval_once(config, train_rows, val_rows, test_rows, device,
                                   run_name=run_name, output_subdir=config.output_dir)
    return test_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images_path", required=True)
    p.add_argument("--labels_path", required=True)
    p.add_argument("--model", default="vit_grit", choices=["vit_grit", "film_resnet50"])
    p.add_argument("--output_dir", default="./outputs")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42,
                   help="training randomness: init, batch order, augmentation")
    p.add_argument("--split_seed", type=int, default=42,
                   help="data partition seed (single split + k-fold folds); independent of --seed")
    p.add_argument("--kfold", type=int, default=0,
                   help="N>1 runs N-fold group-stratified CV; 0 = single split")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    config = Config(
        images_path=args.images_path,
        labels_path=args.labels_path,
        model_name=args.model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        random_seed=args.seed,
        split_seed=args.split_seed,
        kfold=args.kfold,
    )
    run(config)
