import torch
import torch.nn as nn
from torchvision import models


class FiLMLayer(nn.Module):
    """Feature-wise Linear Modulation: output = gamma(cond) * x + beta(cond)."""

    def __init__(self, num_features: int, condition_dim: int):
        super().__init__()
        self.gamma = nn.Linear(condition_dim, num_features)
        self.beta = nn.Linear(condition_dim, num_features)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        g = self.gamma(cond).unsqueeze(-1).unsqueeze(-1)
        b = self.beta(cond).unsqueeze(-1).unsqueeze(-1)
        return g * x + b


class FiLMResNet50(nn.Module):
    """
    ResNet50 with FiLM conditioning applied at layer2, layer3, and layer4.

    The grit embedding is computed once and broadcast into three FiLM layers
    that modulate the CNN feature maps before global pooling.
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

        self.grit_embedding = nn.Sequential(
            nn.Embedding(num_grit_classes, embedding_dim),
            nn.Flatten(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

        base = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
        self.conv1   = base.conv1
        self.bn1     = base.bn1
        self.relu    = base.relu
        self.maxpool = base.maxpool
        self.layer1  = base.layer1
        self.layer2  = base.layer2
        self.layer3  = base.layer3
        self.layer4  = base.layer4
        self.avgpool = base.avgpool

        # ResNet50 channel dims: layer2=512, layer3=1024, layer4=2048
        self.film2 = FiLMLayer(512,  embedding_dim)
        self.film3 = FiLMLayer(1024, embedding_dim)
        self.film4 = FiLMLayer(2048, embedding_dim)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 256),
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
        cond = self.grit_embedding(grit_idx)   # (B, 64)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.film2(self.layer2(x), cond)
        x = self.film3(self.layer3(x), cond)
        x = self.film4(self.layer4(x), cond)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)               # (B, 2048)
        return self.classifier(x)             # (B, 3)
