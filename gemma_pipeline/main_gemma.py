# main_gemma.py
from __future__ import annotations
# --- path shim: allow importing from project root and this folder ---
import os, sys
ROOT = os.path.dirname(os.path.abspath(__file__))          # .../IE4483/Gemma
PROJECT_ROOT = os.path.dirname(ROOT)                       # .../IE4483
for p in (ROOT, PROJECT_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

import argparse, json
from typing import Dict, Tuple, List
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from transformers.utils import logging as hf_logging

# Prefer data_gemma, fall back to data
try:
    from data_gemma import load_train_json, load_test_json, clean_corpus, train_val_split
except Exception:
    from data import load_train_json, load_test_json, clean_corpus, train_val_split

# Prefer eval_tfidf helpers; fall back to eval
try:
    from eval_gemma import evaluate_and_plot, compute_metrics, write_submission
except Exception:
    from eval import evaluate_and_plot, compute_metrics, write_submission

from train_gemma import TrainerGemma

# ---------------- Datasets / Loaders ----------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int] | None):
        self.texts = texts
        self.labels = labels

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        if self.labels is None:
            return {"text": self.texts[idx], "label": None}
        return {"text": self.texts[idx], "label": int(self.labels[idx])}

def make_text_loader(
    texts: List[str],
    labels: List[int] | None,
    tokenizer: AutoTokenizer,
    max_length: int = 256,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
):
    ds = TextDataset(texts, labels)

    def collate(batch):
        batch_texts = [b["text"] for b in batch]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if batch[0]["label"] is None:
            return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": None}
        labels_t = torch.tensor([b["label"] for b in batch], dtype=torch.long)
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels_t}

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        pin_memory=True,
        num_workers=num_workers,
    )

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("Gemma-like embedding classifier | Split-first JOINT Optuna (backbone + head) with inner K-Fold")

    # IO
    ap.add_argument("--train_path", type=str, default="train.json")
    ap.add_argument("--test_path", type=str, default="test.json")
    ap.add_argument("--submission_out", type=str, default="submission_gemma.csv")
    ap.add_argument("--out_dir", type=str, default="gemma_eval")

    # Cleaning / split
    ap.add_argument("--do_clean", action="store_true")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # Model
    ap.add_argument("--model_name", type=str, default="intfloat/e5-base-v2")
    ap.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"])

    # Tuning search spaces (joint with Optuna)
    ap.add_argument("--hidden_candidates", type=str, default="256,512")
    ap.add_argument("--dropout_candidates", type=str, default="0.15,0.2,0.3,0.4")
    ap.add_argument("--init_candidates", type=str, default="xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal")
    ap.add_argument("--max_length_candidates", type=str, default="128,256")
    ap.add_argument("--freeze_epochs_candidates", type=str, default="0,1")  # 0=full FT; 1=frozen 1 epoch then unfreeze

    # LR/WD ranges
    ap.add_argument("--lr_backbone_min", type=float, default=1e-5)
    ap.add_argument("--lr_backbone_max", type=float, default=1e-4)
    ap.add_argument("--lr_head_min", type=float, default=5e-5)
    ap.add_argument("--lr_head_max", type=float, default=5e-3)
    ap.add_argument("--wd_zero_ok", action="store_true")
    ap.add_argument("--wd_min", type=float, default=1e-6)
    ap.add_argument("--wd_max", type=float, default=1e-2)

    # Training (used in folds & final)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--class_weighting", action="store_true")
    ap.add_argument("--num_workers", type=int, default=0)

    # Multi-GPU
    ap.add_argument("--multi_gpu", action="store_true", help="Use nn.DataParallel if multiple GPUs are available")

    # Optuna
    ap.add_argument("--n_trials", type=int, default=25)
    ap.add_argument("--timeout", type=int, default=0, help="seconds; 0 = no timeout")
    ap.add_argument("--inner_k", type=int, default=3, help="K folds inside the TRAIN split")

    # Actions
    ap.add_argument("--retrain_full", action="store_true", help="Retrain on train+val before test prediction")
    ap.add_argument("--make_submission", action="store_true")
    return ap.parse_args()

# -------------- utils ---------------
def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _compute_class_weights(y: List[int]) -> torch.Tensor:
    counts = np.bincount(np.array(y))
    w = (counts.sum() / counts).astype(np.float32)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)

# -------------- Optuna -------------
def _optuna_joint_tune(
    texts_tr: List[str],
    y_tr: List[int],
    args,
    model_name: str,
    pooling: str,
) -> Tuple[Dict, Dict]:
    try:
        import optuna
    except ImportError as e:
        raise SystemExit("Optuna is not installed. Please run: pip install optuna") from e
    from sklearn.model_selection import StratifiedKFold

    hidden_candidates = [int(x) for x in args.hidden_candidates.split(",")]
    dropout_candidates = [float(x) for x in args.dropout_candidates.split(",")]
    init_candidates = [x.strip() for x in args.init_candidates.split(",")]
    maxlen_candidates = [int(x) for x in args.max_length_candidates.split(",")]
    freeze_candidates = [int(x) for x in args.freeze_epochs_candidates.split(",")]

    class_weights_global = _compute_class_weights(y_tr) if args.class_weighting else None

    print("\n=== [Stage 2] Joint Optuna tuning (backbone + head) on TRAIN only ===")
    print(f"Trials: {args.n_trials} | inner_k: {args.inner_k} | seed: {args.seed}")

    hf_logging.set_verbosity_error()
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    except Exception as e:
        raise SystemExit(
            f"\n[Config error] `--model_name {model_name}` was not found on Hugging Face.\n"
            "Use a valid encoder like:\n"
            "  - intfloat/e5-base-v2\n"
            "  - sentence-transformers/all-MiniLM-L6-v2\n"
            "  - BAAI/bge-base-en-v1.5\n"
            "Or pass a local checkpoint directory.\n\n"
            f"Original error: {repr(e)}"
        )

    def objective(trial: "optuna.trial.Trial") -> float:
        # ---- Sample trial params ----
        hidden      = trial.suggest_categorical("hidden", hidden_candidates)
        dropout     = trial.suggest_categorical("dropout", dropout_candidates)
        init_scheme = trial.suggest_categorical("init_scheme", init_candidates)
        max_length  = trial.suggest_categorical("max_length", maxlen_candidates)
        freeze_ep   = trial.suggest_categorical("freeze_backbone_epochs", freeze_candidates)

        lr_backbone = trial.suggest_float("lr_backbone", args.lr_backbone_min, args.lr_backbone_max, log=True)
        lr_head     = trial.suggest_float("lr_head",     args.lr_head_min,     args.lr_head_max,     log=True)

        if args.wd_zero_ok:
            wd_choice = trial.suggest_categorical("wd_choice", ["zero", "log"])
            weight_decay = 0.0 if wd_choice == "zero" else trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)
        else:
            weight_decay = trial.suggest_float("weight_decay", args.wd_min, args.wd_max, log=True)

        # ---- Inner CV on TRAIN only ----
        skf = StratifiedKFold(n_splits=args.inner_k, shuffle=True, random_state=args.seed)
        f1s = []

        for fold_id, (idx_tr, idx_va) in enumerate(skf.split(np.zeros(len(y_tr)), y_tr), start=1):
            x_tr_fold = [texts_tr[i] for i in idx_tr]
            y_tr_fold = [y_tr[i] for i in idx_tr]
            x_va_fold = [texts_tr[i] for i in idx_va]
            y_va_fold = [y_tr[i] for i in idx_va]

            # Loaders (tokenize at collate)
            tr_loader = make_text_loader(
                x_tr_fold, y_tr_fold, tokenizer,
                max_length=max_length, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
            )
            va_loader = make_text_loader(
                x_va_fold, y_va_fold, tokenizer,
                max_length=max_length, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
            )

            cw = _compute_class_weights(y_tr_fold) if args.class_weighting else class_weights_global

            trainer = TrainerGemma(
                model_name=model_name,
                hidden=hidden, dropout=dropout, use_layernorm=True,
                init_scheme=init_scheme, pooling=pooling,
                lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay,
                max_epochs=args.epochs, patience=args.patience,
                class_weights=cw,
                freeze_backbone_epochs=freeze_ep,
                desc_prefix=f"[T{trial.number:03d}|F{fold_id}/{args.inner_k}] ",
                multi_gpu=args.multi_gpu,
            )
            trainer.fit(tr_loader, va_loader)
            va_prob = trainer.predict_proba(va_loader)
            metrics = compute_metrics(y_va_fold, va_prob)
            f1s.append(metrics["macro_f1"])

            # Report/prune
            trial.report(float(np.mean(f1s)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return float(np.mean(f1s))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=args.seed))
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout if args.timeout > 0 else None, show_progress_bar=True)

    print("\n=== Best trial (mean inner-CV macro-F1) ===")
    print(f"Value: {study.best_value:.4f}")
    print(json.dumps(study.best_trial.params, indent=2))

    return study.best_trial.params, {"best_value": study.best_value}

# --------------- main ---------------
def main():
    args = parse_args(); set_seed(args.seed)

    print("=== [Stage 0] Loading data ===")
    texts, labels = load_train_json(args.train_path)
    test_texts = load_test_json(args.test_path)

    if args.do_clean:
        print("... cleaning text")
        texts = clean_corpus(texts); test_texts = clean_corpus(test_texts)

    print("=== [Stage 1] Outer stratified split (hold-out) ===")
    x_tr, x_va, y_tr, y_va = train_val_split(texts, labels, val_size=args.val_size, seed=args.seed)
    print(f"... train={len(x_tr)} | val={len(x_va)}  (seed={args.seed}, val_size={args.val_size})")

    # ---- Joint Optuna (backbone + head) on TRAIN ONLY ----
    best_params, _ = _optuna_joint_tune(x_tr, y_tr, args, model_name=args.model_name, pooling=args.pooling)

    # Unpack best
    hidden       = int(best_params["hidden"])
    dropout      = float(best_params["dropout"])
    init_scheme  = best_params["init_scheme"]
    max_length   = int(best_params["max_length"])
    freeze_ep    = int(best_params.get("freeze_backbone_epochs", 0))
    lr_backbone  = float(best_params["lr_backbone"])
    lr_head      = float(best_params["lr_head"])
    weight_decay = float(0.0 if best_params.get("wd_choice", None) == "zero" else best_params.get("weight_decay", 0.0))

    # ---- Tokenizer and loaders for final train/eval ----
    print("\n=== [Stage 3] Tokenizer & loaders (TRAIN/VAL/TEST) ===")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tr_loader = make_text_loader(x_tr, y_tr, tokenizer, max_length=max_length, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
    va_loader = make_text_loader(x_va, y_va, tokenizer, max_length=max_length, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    te_loader = make_text_loader(test_texts, None, tokenizer, max_length=max_length, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # ---- Final train on TRAIN; eval on VAL ----
    print("\n=== [Stage 4] Train final model on TRAIN; evaluate on VAL ===")
    cw = _compute_class_weights(y_tr) if args.class_weighting else None
    trainer = TrainerGemma(
        model_name=args.model_name,
        hidden=hidden, dropout=dropout, use_layernorm=True,
        init_scheme=init_scheme, pooling=args.pooling,
        lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay,
        max_epochs=args.epochs, patience=args.patience,
        class_weights=cw,
        freeze_backbone_epochs=freeze_ep,
        desc_prefix="[FINAL] ",
        multi_gpu=args.multi_gpu,
    )
    fit_info = trainer.fit(tr_loader, va_loader)
    best_epoch = int(fit_info.get("best_epoch", args.epochs))
    print(f"[FINAL] Best epoch on VAL: {best_epoch}")

    print("\n=== [Stage 5] Validation metrics & plots ===")
    val_prob = trainer.predict_proba(va_loader)
    metrics = evaluate_and_plot(y_va, val_prob, out_dir=args.out_dir, prefix="val")
    print(json.dumps(metrics, indent=2))

    # ---- Optional: retrain on TRAIN+VAL for submission (exact best_epoch) ----
    if args.retrain_full or args.make_submission:
        print(f"\n=== [Stage 6] Retrain on TRAIN+VAL for {best_epoch} epoch(s) & prepare submission ===")
        x_full = x_tr + x_va
        y_full = y_tr + y_va

        tr_full = make_text_loader(x_full, y_full, tokenizer, max_length=max_length, batch_size=args.batch_size, shuffle=True,  num_workers=args.num_workers)
        te_full = make_text_loader(test_texts, None, tokenizer, max_length=max_length, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        trainer_full = TrainerGemma(
            model_name=args.model_name,
            hidden=hidden, dropout=dropout, use_layernorm=True,
            init_scheme=init_scheme, pooling=args.pooling,
            lr_backbone=lr_backbone, lr_head=lr_head, weight_decay=weight_decay,
            max_epochs=best_epoch, patience=args.patience,
            class_weights=_compute_class_weights(y_full) if args.class_weighting else None,
            freeze_backbone_epochs=freeze_ep,
            desc_prefix="[FULL] ",
            multi_gpu=args.multi_gpu,
        )
        trainer_full.fit(tr_full, None)

        if args.make_submission:
            print("\n=== [Stage 7] Writing submission ===")
            test_pred = trainer_full.predict(te_full)
            write_submission(test_pred, path=args.submission_out)

if __name__ == "__main__":
    main()
