# models_tfidf.py
from __future__ import annotations
import torch
import torch.nn as nn

def _apply_init(module: nn.Module, scheme: str):
    if isinstance(module, nn.Linear):
        if scheme == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif scheme == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        elif scheme == "kaiming_uniform":
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        elif scheme == "kaiming_normal":
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        else:
            nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)

class FFNNClassifier(nn.Module):
    """Two-layer MLP for dense TF-IDF features."""
    def __init__(
        self,
        in_dim: int,
        hidden: int = 512,
        dropout: float = 0.3,
        use_layernorm: bool = True,
        num_classes: int = 2,
        init_scheme: str = "xavier_uniform",
    ):
        super().__init__()
        layers = []
        if use_layernorm:
            layers.append(nn.LayerNorm(in_dim))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, hidden))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, num_classes))
        self.net = nn.Sequential(*layers)
        self.init_scheme = init_scheme
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        _apply_init(m, self.init_scheme)

    def forward(self, x):
        return self.net(x)
