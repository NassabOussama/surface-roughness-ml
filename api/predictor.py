from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from config import CLASSES  # resolved from new/ root


EVAL_TRANSFORM = T.Compose([
    T.Resize(int(224 * 1.1)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device)

    from models.vit_grit import ViTGRiT
    from models.film_resnet50 import FiLMResNet50

    kwargs = dict(
        num_grit_classes=ckpt["num_grit_classes"],
        embedding_dim=ckpt["embedding_dim"],
        dropout_rate=ckpt["dropout_rate"],
        num_classes=ckpt["num_classes"],
        pretrained=False,
    )
    model_cls = ViTGRiT if ckpt["model_name"] == "vit_grit" else FiLMResNet50
    model = model_cls(**kwargs)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    grit_to_idx = {g: i for i, g in enumerate(ckpt["grit_values"])}
    return model, grit_to_idx


class Predictor:
    """
    Production inference wrapper. Loads a checkpoint once and exposes predict().
    """

    def __init__(self, checkpoint_path: str, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.grit_to_idx = _load_model(checkpoint_path, self.device)

    def _prepare_image(self, image: Union[str, Path, Image.Image]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        return EVAL_TRANSFORM(image).unsqueeze(0).to(self.device)

    def _prepare_grit(self, grit: int) -> torch.Tensor:
        if grit not in self.grit_to_idx:
            raise ValueError(f"Unknown grit {grit}. Valid: {sorted(self.grit_to_idx)}")
        return torch.tensor([self.grit_to_idx[grit]], dtype=torch.long, device=self.device)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, Image.Image], grit: int) -> Dict:
        img_t = self._prepare_image(image)
        grit_t = self._prepare_grit(grit)

        logits = self.model(img_t, grit_t)           # (1, 3)
        probs = torch.softmax(logits, dim=1)[0]      # (3,)
        class_idx = int(probs.argmax())

        return {
            "label": CLASSES[class_idx],
            "confidence": float(probs[class_idx]),
            "probabilities": {c: float(probs[i]) for i, c in enumerate(CLASSES)},
        }
