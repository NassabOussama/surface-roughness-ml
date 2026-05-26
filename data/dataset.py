import os
import csv
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

from config import Config, ra_to_label


def load_labels(labels_path: str) -> List[dict]:
    rows = []
    with open(labels_path, encoding="utf-8-sig") as f:
        content = f.read().replace("\r", "")
        reader = csv.DictReader(content.splitlines(), delimiter=";")
        for row in reader:
            ra = float(row["Ra_Moyenne"].replace(",", "."))
            rows.append({
                "numero": int(row["Numero"]),
                "face": row["Face"],
                "grit": int(row["Grit_Utilise"]),
                "ra": ra,
                "label": ra_to_label(ra),
                "group_id": f"{row['Numero']}_{row['Face']}",
            })
    return rows


class RoughnessDataset(Dataset):
    """
    Each physical sample has 4 images (angles). Each image is an independent
    training example; all 4 share the same label. Groups are kept together
    during splitting to prevent data leakage.
    """

    def __init__(
        self,
        rows: List[dict],
        images_path: str,
        grit_to_idx: Dict[int, int],
        transform: Optional[T.Compose] = None,
    ):
        self.images_path = images_path
        self.grit_to_idx = grit_to_idx
        self.transform = transform

        self.samples: List[dict] = []
        for row in rows:
            base = f"{row['numero']}{row['face']}"
            for i in range(1, 5):
                img_path = os.path.join(images_path, f"{base}_{i}.png")
                if os.path.exists(img_path):
                    self.samples.append({
                        "image_path": img_path,
                        "grit_idx": grit_to_idx[row["grit"]],
                        "label": row["label"],
                        "group_id": row["group_id"],
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        image = Image.open(s["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        grit_idx = torch.tensor(s["grit_idx"], dtype=torch.long)
        label = torch.tensor(s["label"], dtype=torch.long)
        return image, grit_idx, label


def group_split(
    rows: List[dict],
    config: Config,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Stratified group split so all 4 images of a sample stay in the same fold.
    Stratification is by grit value to preserve distribution.
    """
    from sklearn.model_selection import train_test_split

    groups = list({r["group_id"]: r for r in rows}.values())
    group_ids = [g["group_id"] for g in groups]
    grits = [g["grit"] for g in groups]

    train_ids, temp_ids, _, temp_grits = train_test_split(
        group_ids, grits,
        test_size=config.val_split + config.test_split,
        random_state=config.random_seed,
        stratify=grits,
    )
    rel_test = config.test_split / (config.val_split + config.test_split)
    val_ids, test_ids = train_test_split(
        temp_ids,
        test_size=rel_test,
        random_state=config.random_seed,
        stratify=temp_grits,
    )

    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    assert not (train_set & val_set) and not (train_set & test_set) and not (val_set & test_set)

    return (
        [r for r in rows if r["group_id"] in train_set],
        [r for r in rows if r["group_id"] in val_set],
        [r for r in rows if r["group_id"] in test_set],
    )
