import torch
import torch.nn as nn
import timm


class ViTGRiT(nn.Module):
    """
    Vision Transformer backbone with grit-value conditioning via concatenation.

    Image features (768-dim CLS token) are concatenated with a learned grit
    embedding (64-dim) before the classification head.
    """

    def __init__(
        self,
        num_grit_classes: int = 10,
        embedding_dim: int = 64,
        dropout_rate: float = 0.3,
        num_classes: int = 3,
        pretrained: bool = True,
    ):
        super().__init__()

        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=pretrained,
            num_classes=0,  # remove classifier head
        )
        vit_dim = self.vit.num_features  # 768

        self.grit_embedding = nn.Sequential(
            nn.Embedding(num_grit_classes, embedding_dim),
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        combined_dim = vit_dim + embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, num_classes),
        )

        self._init_head()

    def _init_head(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, grit_idx: torch.Tensor) -> torch.Tensor:
        img_feat = self.vit(x)                              # (B, 768)
        grit_emb = self.grit_embedding(grit_idx)           # (B, 64)
        combined = torch.cat([img_feat, grit_emb], dim=1)  # (B, 832)
        return self.classifier(combined)                    # (B, 3)
