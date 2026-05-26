import os
from dataclasses import dataclass, field
from typing import Tuple


CLASSES = ("Smooth", "Medium", "Rough")
RA_THRESHOLDS = (0.653, 0.960)  # 33rd / 67th percentile of training Ra_Moyenne


def ra_to_label(ra: float) -> int:
    """Map Ra value (μm) to class index: 0=Smooth, 1=Medium, 2=Rough."""
    if ra < RA_THRESHOLDS[0]:
        return 0
    if ra < RA_THRESHOLDS[1]:
        return 1
    return 2


@dataclass
class Config:
    # === PATHS ===
    images_path: str = ""
    labels_path: str = ""
    output_dir: str = "./outputs"

    # === IMAGE ===
    input_size: int = 224

    # === GRIT ===
    grit_values: Tuple[int, ...] = (60, 80, 100, 120, 150, 180, 240, 320, 400, 600)

    # === MODEL ===
    model_name: str = "vit_grit"       # "vit_grit" | "film_resnet50"
    embedding_dim: int = 64
    dropout_rate: float = 0.3
    num_classes: int = 3

    # === TRAINING ===
    batch_size: int = 16
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4

    # === DATA SPLIT ===
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42

    # === EARLY STOPPING ===
    patience: int = 15
    min_delta: float = 1e-4

    # === SCHEDULER ===
    scheduler_T0: int = 10
    scheduler_T_mult: int = 2

    @property
    def num_grit_classes(self) -> int:
        return len(self.grit_values)

    @property
    def grit_to_idx(self):
        return {g: i for i, g in enumerate(self.grit_values)}

    def __post_init__(self):
        assert abs(self.train_split + self.val_split + self.test_split - 1.0) < 1e-6
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
