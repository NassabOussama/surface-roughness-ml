"""
Generate publication-quality figures from the rigorous benchmark results.

Reads metrics straight from the saved fold/seed checkpoints (which store the
confusion matrix + per-class F1) and pulls per-epoch training curves from the
local MLflow store (sqlite:///mlflow.db). Writes PNGs to assets/.

Note: depends on the local (gitignored) outputs/ and mlflow.db, so it only
reproduces after the training runs have been executed.
"""
import glob
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

CLASSES = ["Smooth", "Medium", "Rough"]
ASSETS = "assets"
KFOLD_DIR = "outputs/rigorous/kfold"
FIXED_DIR = "outputs/rigorous/fixed_split"
COLORS = {"ViT-GRiT": "#4C72B0", "FiLM-ResNet50": "#DD8452"}
DIRNAME = {"ViT-GRiT": "vit_grit", "FiLM-ResNet50": "film_resnet50"}
MODELS = ["ViT-GRiT", "FiLM-ResNet50"]

os.makedirs(ASSETS, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
})


def _load_metrics(pth_glob):
    paths = glob.glob(pth_glob)
    if not paths:
        raise FileNotFoundError(pth_glob)
    d = torch.load(paths[0], map_location="cpu", weights_only=False)
    return d["metrics"]


def load_cv_folds(model_dir):
    folds = {}
    for fold in sorted(glob.glob(os.path.join(model_dir, "fold*"))):
        idx = int(os.path.basename(fold).replace("fold", ""))
        m = _load_metrics(os.path.join(fold, "*.pth"))
        folds[idx] = {
            "acc": float(m["accuracy"]),
            "f1": float(m["f1"]),
            "cm": np.asarray(m["confusion_matrix"], dtype=int),
        }
    return folds


def load_fixed_seeds(model_key, seeds=(42, 123, 2024)):
    accs, f1s = [], []
    for s in seeds:
        m = _load_metrics(f"{FIXED_DIR}/{DIRNAME[model_key]}_seed{s}/*.pth")
        accs.append(float(m["accuracy"]))
        f1s.append(float(m["f1"]))
    return np.array(accs), np.array(f1s)


# ---- gather all data (no hardcoded metrics) -------------------------------
cv = {m: load_cv_folds(f"{KFOLD_DIR}/{DIRNAME[m]}") for m in MODELS}
fixed = {m: load_fixed_seeds(m) for m in MODELS}


# ---------------------------------------------------------------- figure (a)
def fig_benchmark():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    metric_idx = {"Accuracy": 0, "Macro-F1": 1}
    groups = ["3-seed fixed split", "5-fold CV"]
    x = np.arange(len(groups))
    w = 0.36

    for ax, (mname, mi) in zip(axes, metric_idx.items()):
        for j, model in enumerate(MODELS):
            fixed_vals = fixed[model][mi]
            cv_vals = np.array([f[("acc", "f1")[mi]] for f in cv[model].values()])
            means = [fixed_vals.mean(), cv_vals.mean()]
            stds = [fixed_vals.std(ddof=1), cv_vals.std(ddof=1)]
            bars = ax.bar(x + (j - 0.5) * w, means, w, yerr=stds, capsize=5,
                          label=model, color=COLORS[model], alpha=0.9,
                          error_kw={"elinewidth": 1.3, "ecolor": "#333"})
            for b, mn, sd in zip(bars, means, stds):
                ax.text(b.get_x() + b.get_width() / 2, mn + sd + 0.012,
                        f"{mn:.3f}\n±{sd:.3f}", ha="center", va="bottom", fontsize=8.5)
        ax.set_xticks(x)
        ax.set_xticklabels(groups)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel(mname)
        ax.set_title(mname)
        ax.axhline(1 / 3, ls=":", c="grey", lw=1)
        ax.text(1.45, 1 / 3 + 0.01, "chance", color="grey", fontsize=8, ha="right")

    axes[0].legend(loc="lower left")
    fig.suptitle("Benchmark comparison — mean ± std across runs",
                 fontsize=14, fontweight="bold", y=1.02)
    out = f"{ASSETS}/benchmark_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure (b)
def _draw_cm(ax, cm, title):
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    ax.set_xticks(range(3)); ax.set_xticklabels(CLASSES)
    ax.set_yticks(range(3)); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    thr = cm.max() / 2
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222", fontsize=12)
    for s in ax.spines.values():
        s.set_visible(True)
    return im


def fig_confusion():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.5})
    for ax, model in zip(axes, MODELS):
        best = max(cv[model].items(), key=lambda kv: kv[1]["acc"])
        idx, m = best
        title = f"{model} — best CV fold (fold {idx})\nacc {m['acc']:.3f}  |  macro-F1 {m['f1']:.3f}"
        im = _draw_cm(ax, m["cm"], title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Confusion matrices — best cross-validation fold per model",
                 fontsize=14, fontweight="bold", y=1.03)
    out = f"{ASSETS}/confusion_matrices.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure (c)
def fig_training_history():
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    client = MlflowClient()
    exp = client.get_experiment_by_name("surface-roughness-classification")
    runs = client.search_runs(exp.experiment_id, max_results=1000)

    def best_cv_run(model_key):
        prefix = f"{DIRNAME[model_key]}_cv5"
        cand = [r for r in runs
                if r.data.tags.get("mlflow.runName", "").startswith(prefix)
                and "_fold" in r.data.tags.get("mlflow.runName", "")
                and r.data.metrics.get("test_acc") is not None]
        return max(cand, key=lambda r: r.data.metrics["test_acc"])

    def history(run_id, key):
        h = sorted(client.get_metric_history(run_id, key), key=lambda x: x.step)
        return [x.step + 1 for x in h], [x.value for x in h]

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, model in enumerate(MODELS):
        run = best_cv_run(model)
        rid = run.info.run_id
        fold = run.data.tags["mlflow.runName"].split("_")[-1]
        c = COLORS[model]

        e, tl = history(rid, "train_loss"); _, vl = history(rid, "val_loss")
        axes[row, 0].plot(e, tl, color=c, lw=1.8, label="train")
        axes[row, 0].plot(e, vl, color=c, lw=1.8, ls="--", label="val")
        axes[row, 0].set_title(f"{model} — Loss (best CV {fold})")
        axes[row, 0].set_xlabel("Epoch"); axes[row, 0].set_ylabel("Loss")
        axes[row, 0].legend()

        e, ta = history(rid, "train_acc"); _, va = history(rid, "val_acc")
        axes[row, 1].plot(e, ta, color=c, lw=1.8, label="train")
        axes[row, 1].plot(e, va, color=c, lw=1.8, ls="--", label="val")
        axes[row, 1].set_title(f"{model} — Accuracy (best CV {fold})")
        axes[row, 1].set_xlabel("Epoch"); axes[row, 1].set_ylabel("Accuracy")
        axes[row, 1].set_ylim(0, 1.0)
        axes[row, 1].legend()

    fig.suptitle("Training history — best cross-validation fold per model",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = f"{ASSETS}/training_history.png"
    fig.savefig(out)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- figure (d)
def fig_per_class():
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.arange(len(CLASSES))
    w = 0.36
    totals = {}
    for j, model in enumerate(MODELS):
        total_cm = sum(f["cm"] for f in cv[model].values())  # 5 folds = all 360 imgs
        totals[model] = total_cm
        recall = np.array([total_cm[i, i] / total_cm[i].sum() for i in range(3)])
        bars = ax.bar(x + (j - 0.5) * w, recall, w, label=model,
                      color=COLORS[model], alpha=0.9)
        for b, r in zip(bars, recall):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.012, f"{r:.3f}",
                    ha="center", va="bottom", fontsize=9)
    n = int(totals[MODELS[0]].sum())
    ax.set_xticks(x); ax.set_xticklabels(CLASSES)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Per-class accuracy (recall)")
    ax.set_title(f"Per-class accuracy — aggregated over 5-fold CV (n={n} images)")
    ax.legend(loc="lower center")
    out = f"{ASSETS}/per_class_accuracy.png"
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    for fn in (fig_benchmark, fig_confusion, fig_training_history, fig_per_class):
        print("wrote", fn())
    print("done")
