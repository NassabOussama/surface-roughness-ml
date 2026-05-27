# CLAUDE.md

Guidance for working in this repository.

## Project overview

Image-based **surface roughness classification** for machined metal surfaces. Given a
microscopy image of a machined surface **plus the abrasive grit value** used during
machining, the model predicts a 3-class roughness category.

The grit value is a real conditioning signal: the same visual texture implies different
roughness depending on the abrasive used, so every model fuses the image features with a
learned grit embedding rather than classifying the image alone.

### Classes and Ra thresholds

Labels are derived from `Ra_Moyenne` (mean arithmetic roughness, μm). Thresholds are the
33rd / 67th percentiles of the training set's `Ra_Moyenne`:

| Class  | Index | Condition (Ra in μm)        |
|--------|-------|-----------------------------|
| Smooth | 0     | `Ra < 0.653`                |
| Medium | 1     | `0.653 <= Ra < 0.960`       |
| Rough  | 2     | `Ra >= 0.960`               |

Defined in `config.py` (`CLASSES`, `RA_THRESHOLDS`, `ra_to_label`).

## Dataset

- **Labels:** `Labelisation_CSV.csv` — `;`-delimited, comma decimal separator, UTF-8 BOM.
  Columns: `Numero;Face;Nom;Grit_Utilise;Ra1;Ra2;Ra3;Ra_Moyenne`. ~90 physical samples.
- **Images:** one directory of PNGs. Each physical sample = `<Numero><Face>` (e.g. `10A`)
  has **4 angle images** named `<Numero><Face>_<1..4>.png` (e.g. `10A_1.png` … `10A_4.png`),
  so ~360 images total. All 4 images of a sample share the same label.
- **Grit values** (10 classes): `60, 80, 100, 120, 150, 180, 240, 320, 400, 600`.
- The real dataset is **kept local** (gitignored). On this setup it lives at
  `old/Images` and `old/Labelisation_CSV.csv`. The `data/full_dataset/` path is also
  gitignored. The dataset is NOT distributed via git — copy it to the machine manually.

### Data splitting (important — prevents leakage)

`data/dataset.py::group_split` does a **stratified group split**: the unit of splitting is
the physical sample (`group_id = "<Numero>_<Face>"`), so all 4 angle images of a sample
stay together in the same fold. Stratification is by grit value to preserve its
distribution. Default split is 70/15/15 (train/val/test), seed 42. Never split at the
image level — that would leak a sample's other angles across folds.

## Models

Both models take `(image, grit_idx)` and output 3-class logits. Selected via
`--model` / `Config.model_name`.

### ViT-GRiT (primary, default: `vit_grit`)

`models/vit_grit.py`. `timm` `vit_base_patch16_224` backbone (classifier removed,
768-d CLS token). The grit index is passed through an embedding MLP (→ 64-d) and
**concatenated** with the image features (768 + 64 = 832) before an MLP head
(832 → 256 → 64 → 3, with BatchNorm + dropout).

### FiLM-ResNet50 (alternative: `film_resnet50`)

`models/film_resnet50.py`. ResNet50 backbone (ImageNet pretrained). The grit embedding
conditions the network via **FiLM** (Feature-wise Linear Modulation): `gamma(cond) * x +
beta(cond)` applied to the feature maps after `layer2`, `layer3`, and `layer4`. FiLM
layers are initialized to identity (gamma=1, beta=0). Head: 2048 → 256 → 64 → 3.

> History: the project began as FiLM-ResNet34, was refactored to ViT-GRiT (see
> `old/REFACTORING_CHANGE_LOG.md`). ViT-GRiT is the current primary model.

## Training

Entry point: `train.py` (`config.py::Config` holds all hyperparameters).

- Optimizer: AdamW (lr 1e-4, weight_decay 1e-4).
- Scheduler: CosineAnnealingWarmRestarts (T_0=10, T_mult=2, eta_min = lr*0.01).
- Loss: CrossEntropyLoss. Gradient clipping at norm 1.0.
- **AMP** (mixed precision) auto-enabled on CUDA.
- Early stopping on val accuracy (patience 15, min_delta 1e-4); best weights restored
  before test evaluation.
- **Decoupled seeds:** `--split_seed` (default 42) controls *only* the train/val/test
  partition; `--seed` controls training randomness (weight-head init, batch order,
  augmentation). Fixing `--split_seed` while varying `--seed` gives multiple runs on the
  *same* test set — the correct setup for comparing models. Note: training is not
  bit-deterministic on GPU (AMP + non-deterministic CUDA kernels), so same-seed runs
  still vary slightly.
- **K-fold CV:** `--kfold N` (N>1) runs N folds of group-stratified CV (groups kept
  together, stratified by grit, partitioned by `split_seed`); each fold carves a val set
  from its train portion for early stopping. Reports fold-averaged mean ± std and writes
  `cv_summary.csv`. `--kfold 0` (default) = single split, unchanged behaviour. Given only
  90 samples, k-fold uses the data far better than a single 56-image test split.
- **MLflow** logging (experiment `surface-roughness-classification`): params, per-epoch
  metrics, plots, and (single-split only) the checkpoint + pickled model. K-fold logs one
  nested run per fold under a parent run with `cv_*_mean`/`cv_*_std`; per-fold heavy model
  artifacts are skipped to avoid duplicating GBs (checkpoints still land on disk).
- Augmentation (`data/transforms.py`): train uses RandomResizedCrop, rotation/flips,
  light color jitter, occasional Gaussian blur; val/test use Resize+CenterCrop. ImageNet
  normalization throughout. Input size 224.

Checkpoints (`save_checkpoint`) store `model_state_dict` plus the metadata needed to
rebuild the model for inference (`model_name`, grit values, dims, metrics).

## Inference / serving

- `evaluate.py` — `evaluate()` (loss, accuracy, macro-F1, per-class F1, confusion matrix)
  and plotting helpers.
- `api/predictor.py` — `Predictor` loads a checkpoint and exposes `predict(image, grit)`
  → `{label, confidence, probabilities}`. Rejects unknown grit values.
- `api/main.py` — FastAPI service: `POST /predict` (image file + grit form field),
  `GET /health`. Checkpoint path via `CHECKPOINT_PATH` env (default `./model.pth`).
- `ui/app.py` — Streamlit front-end that calls the API (`API_URL` env).
- `docker-compose.yml` — builds `api` (port 8000) and `ui` (port 8501). Mount a trained
  checkpoint to `/app/model.pth`.

## Commands

```bash
# Train (primary model)
python train.py \
    --images_path old/Images \
    --labels_path old/Labelisation_CSV.csv \
    --model vit_grit \
    --output_dir ./outputs

# Train the alternative model
python train.py --images_path old/Images --labels_path old/Labelisation_CSV.csv --model film_resnet50

# Multiple training seeds on the SAME fixed test set (for model comparison)
python train.py ... --model vit_grit --split_seed 42 --seed 123

# 5-fold group-stratified cross-validation
python train.py ... --model vit_grit --kfold 5 --split_seed 42

# Useful train flags: --batch_size --epochs --lr --seed --split_seed --kfold

# Serve API (needs a checkpoint at ./model.pth or CHECKPOINT_PATH)
CHECKPOINT_PATH=outputs/<checkpoint>.pth uvicorn api.main:app --host 0.0.0.0 --port 8000

# Streamlit UI (API must be running)
API_URL=http://localhost:8000 streamlit run ui/app.py

# Full stack via Docker
docker-compose up --build
```

## Environment notes

- Training runs on a Windows PC with an **RTX 4070** (CUDA). torch is the cu121 build.
- Dependencies in `requirements.txt`. Key: torch/torchvision, timm, mlflow,
  scikit-learn, fastapi/uvicorn, streamlit, Pillow, matplotlib, numpy.
- `outputs/`, `mlruns/`, `*.pth`, `old/`, and `data/full_dataset/` are gitignored.
