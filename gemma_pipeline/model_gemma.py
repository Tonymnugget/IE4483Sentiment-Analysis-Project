# models_gemma.py
from __future__ import annotations
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

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
        else:  # default fallback
            nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, T, H]; attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)  # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
    return summed / counts

class SentenceEncoderClassifier(nn.Module):
    """
    Transformer backbone (Gemma embedding model or equivalent HF model) + 2-layer MLP head.
    """
    def __init__(
        self,
        model_name: str = "embeddinggemma-300M",
        hidden: int = 512,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        init_scheme: str = "xavier_uniform",
        pooling: str = "mean",
        num_classes: int = 2,
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        emb_dim = getattr(self.config, "hidden_size", None)
        if emb_dim is None:
            # Fallback for models exposing sentence embedding directly
            emb_dim = self.backbone.config.hidden_size

        self.pooling = pooling
        head = []
        if use_layernorm:
            head.append(nn.LayerNorm(emb_dim))
        head.append(nn.Dropout(dropout))
        head.append(nn.Linear(emb_dim, hidden))
        head.append(nn.ReLU(inplace=True))
        head.append(nn.Dropout(dropout))
        head.append(nn.Linear(hidden, num_classes))
        self.head = nn.Sequential(*head)

        # Initialize head
        self.init_scheme = init_scheme
        self.head.apply(lambda m: _apply_init(m, self.init_scheme))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        if self.pooling == "mean":
            return mean_pool(out.last_hidden_state, attention_mask)
        elif self.pooling == "cls":
            return out.last_hidden_state[:, 0]
        else:
            # default: mean
            return mean_pool(out.last_hidden_state, attention_mask)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        emb = self.encode(input_ids, attention_mask)
        return self.head(emb)
