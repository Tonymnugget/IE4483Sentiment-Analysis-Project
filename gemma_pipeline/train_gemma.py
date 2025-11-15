# train_gemma.py
from __future__ import annotations
from typing import Optional, Dict
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model_gemma import SentenceEncoderClassifier

class TrainerGemma:
    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        hidden: int = 512,
        dropout: float = 0.2,
        use_layernorm: bool = True,
        init_scheme: str = "xavier_uniform",
        pooling: str = "mean",
        lr_backbone: float = 5e-5,
        lr_head: float = 1e-3,
        weight_decay: float = 0.0,
        max_epochs: int = 10,
        patience: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        freeze_backbone_epochs: int = 0,
        device: Optional[str] = None,
        desc_prefix: str = "",
        multi_gpu: bool = False,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build the base model first (so optimizer can see param groups)
        base_model = SentenceEncoderClassifier(
            model_name=model_name,
            hidden=hidden,
            dropout=dropout,
            use_layernorm=use_layernorm,
            init_scheme=init_scheme,
            pooling=pooling,
            num_classes=2,
        ).to(self.device)

        # Optimizer with separate LR for backbone vs head
        self.optimizer = AdamW(
            [
                {"params": base_model.backbone.parameters(), "lr": lr_backbone, "weight_decay": weight_decay},
                {"params": base_model.head.parameters(),     "lr": lr_head,     "weight_decay": weight_decay},
            ]
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max_epochs)
        self.max_epochs = max_epochs
        self.patience = patience
        self.class_weights = class_weights.to(self.device) if class_weights is not None else None
        self.freeze_backbone_epochs = max(0, int(freeze_backbone_epochs))
        self.desc_prefix = desc_prefix

        # Optional DataParallel (single-process, multi-GPU)
        self.is_parallel = (multi_gpu and torch.cuda.is_available() and torch.cuda.device_count() > 1)
        if self.is_parallel:
            print(f"{self.desc_prefix}Using DataParallel on {torch.cuda.device_count()} GPUs")
            self.model = nn.DataParallel(base_model)
        else:
            self.model = base_model

    def _underlying(self) -> SentenceEncoderClassifier:
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _set_backbone_trainable(self, trainable: bool):
        m = self._underlying()
        for p in m.backbone.parameters():
            p.requires_grad = trainable

    def _step(self, batch, train: bool = True):
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask)

        if "labels" in batch and batch["labels"] is not None:
            y = batch["labels"].to(self.device, non_blocking=True)
            loss = F.cross_entropy(logits, y, weight=self.class_weights)
        else:
            # inference path (no labels)
            loss = logits.new_tensor(0.0)

        if train:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._underlying().parameters(), 1.0)
            self.optimizer.step()

        with torch.no_grad():
            if "labels" in batch and batch["labels"] is not None:
                preds = logits.argmax(dim=-1)
                acc = (preds == y).float().mean().item()
            else:
                acc = 0.0
        return float(loss.item()), acc

    def fit(self, train_loader, val_loader: Optional[torch.utils.data.DataLoader] = None) -> Dict:
        """
        Returns:
            If val_loader is None:
                {"epochs_trained": <int>}
            Else:
                {"best_epoch": <int>, "best_val_acc": <float>, "epochs_trained": <int>}
        """
        # Start frozen or not depending on freeze_backbone_epochs
        self._set_backbone_trainable(self.freeze_backbone_epochs == 0)

        # --------- No-validation path (Stage 6 retrain) ----------
        if val_loader is None:
            for epoch in range(1, self.max_epochs + 1):
                # Unfreeze after freeze_backbone_epochs
                if epoch == (self.freeze_backbone_epochs + 1):
                    self._set_backbone_trainable(True)

                self.model.train()
                tr_loss, tr_acc, n = 0.0, 0.0, 0
                pbar = tqdm(train_loader, desc=f"{self.desc_prefix}Epoch {epoch:02d} [train]", leave=False, dynamic_ncols=True)
                for batch in pbar:
                    loss, acc = self._step(batch, train=True)
                    bsz = batch["input_ids"].size(0)
                    tr_loss += loss * bsz
                    tr_acc  += acc  * bsz
                    n += bsz
                    pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
                tr_loss /= max(n, 1); tr_acc /= max(n, 1)
                self.scheduler.step()
                print(f"[{self.desc_prefix}Epoch {epoch:02d}] train_only loss={tr_loss:.4f} acc={tr_acc:.4f}")
            return {"epochs_trained": self.max_epochs}

        # --------- Train + Val path (Stage 4 with early stopping) ----------
        best_val_acc = -1.0
        best_state = None
        best_epoch = 0
        epochs_no_improve = 0
        total_epochs_run = 0

        for epoch in range(1, self.max_epochs + 1):
            total_epochs_run = epoch

            # Unfreeze after freeze_backbone_epochs
            if epoch == (self.freeze_backbone_epochs + 1):
                self._set_backbone_trainable(True)

            # Train
            self.model.train()
            tr_loss, tr_acc, n = 0.0, 0.0, 0
            pbar = tqdm(train_loader, desc=f"{self.desc_prefix}Epoch {epoch:02d} [train]", leave=False, dynamic_ncols=True)
            for batch in pbar:
                loss, acc = self._step(batch, train=True)
                bsz = batch["input_ids"].size(0)
                tr_loss += loss * bsz
                tr_acc  += acc  * bsz
                n += bsz
                pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
            tr_loss /= max(n, 1); tr_acc /= max(n, 1)

            # Val
            self.model.eval()
            va_loss, va_acc, n = 0.0, 0.0, 0
            pbar = tqdm(val_loader, desc=f"{self.desc_prefix}Epoch {epoch:02d} [val]  ", leave=False, dynamic_ncols=True)
            with torch.no_grad():
                for batch in pbar:
                    loss, acc = self._step(batch, train=False)
                    bsz = batch["input_ids"].size(0)
                    va_loss += loss * bsz
                    va_acc  += acc  * bsz
                    n += bsz
                    pbar.set_postfix(loss=f"{loss:.4f}", acc=f"{acc:.4f}")
            va_loss /= max(n, 1); va_acc /= max(n, 1)

            self.scheduler.step()
            print(f"[{self.desc_prefix}Epoch {epoch:02d}] train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            if va_acc > best_val_acc:
                best_val_acc = va_acc
                best_epoch = epoch
                # keep a CPU copy of the (possibly DP) model state
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"[{self.desc_prefix}] Early stopping at epoch {epoch}. Best epoch={best_epoch} val_acc={best_val_acc:.4f}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return {"best_epoch": best_epoch, "best_val_acc": float(best_val_acc), "epochs_trained": total_epochs_run}

    def predict_proba(self, data_loader):
        self.model.eval()
        preds = []
        pbar = tqdm(data_loader, desc=f"{self.desc_prefix}Predict", leave=False, dynamic_ncols=True)
        with torch.no_grad():
            for batch in pbar:
                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                prob = torch.softmax(logits, dim=-1)
                preds.append(prob.cpu())
        return torch.cat(preds, dim=0).numpy()

    def predict(self, data_loader):
        proba = self.predict_proba(data_loader)
        return proba.argmax(axis=-1)
