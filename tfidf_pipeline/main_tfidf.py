# main_tfidf.py
from __future__ import annotations
import argparse, json, os
from typing import Dict, Tuple, List
import numpy as np
from tqdm.auto import tqdm
import torch

from data import (
    load_train_json, load_test_json, clean_corpus, train_val_split, make_loader
)
from features_tfidf import TfidfFeaturizer
from train_tfidf import TrainerTFIDF
from eval_tfidf import evaluate_and_plot, compute_metrics, write_submission

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser("TF-IDF + FFNN | Split-first JOINT Optuna (features + head) with inner K-Fold")

    # IO
    ap.add_argument("--train_path", type=str, default="train.json")
    ap.add_argument("--test_path", type=str, default="test.json")
    ap.add_argument("--vectorizer_out", type=str, default="tfidf.pkl")
    ap.add_argument("--submission_out", type=str, default="submission.csv")
    ap.add_argument("--out_dir", type=str, default="tfidf_eval")

    # Cleaning / split
    ap.add_argument("--do_clean", action="store_true")
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)

    # JOINT tuning search spaces (features + head)
    ap.add_argument("--mf_candidates", type=str, default="10000,15000,20000,25000,30000,40000")
    ap.add_argument("--ng_candidates", type=str, default="1,2,3,4")
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--max_df", type=float, default=0.9)

    ap.add_argument("--hidden_candidates", type=str, default="256,512")
    ap.add_argument("--dropout_candidates", type=str, default="0.15,0.2,0.3,0.4")
    ap.add_argument("--init_candidates", type=str, default="xavier_uniform,xavier_normal,kaiming_uniform,kaiming_normal")
    ap.add_argument("--lr_min", type=float, default=1e-4)
    ap.add_argument("--lr_max", type=float, default=1e-3)
    ap.add_argument("--wd_zero_ok", action="store_true")
    ap.add_argument("--wd_min", type=float, default=1e-6)
    ap.add_argument("--wd_max", type=float, default=1e-2)

    # Training (used in folds & final)
    ap.add_argument("--use_layernorm", action="store_true")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--class_weighting", action="store_true")

    # Optuna
    ap.add_argument("--n_trials", type=int, default=40)
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

def _train_eval_once(
    Xtr: np.ndarray, y_tr: List[int],
    Xva: np.ndarray, y_va: List[int],
    args,
    hidden: int, dropout: float, init_scheme: str,
    lr: float, weight_decay: float,
    class_weights: torch.Tensor | None,
    in_dim: int,
    desc_prefix: str = "",
) -> Dict:
    tr_loader = make_loader(Xtr, y_tr, batch_size=args.batch_size, shuffle=True)
    va_loader = make_loader(Xva, y_va, batch_size=args.batch_size, shuffle=False)
    trainer = TrainerTFIDF(
        in_dim=in_dim,
        hidden=hidden,
        dropout=dropout,
        use_layernorm=args.use_layernorm,
        init_scheme=init_scheme,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=args.epochs,
        patience=args.patience,
        class_weights=class_weights if args.class_weighting else None,
        desc_prefix=desc_prefix,
    )
    trainer.fit(tr_loader, va_loader)
    va_prob = trainer.predict_proba(va_loader)
    return compute_metrics(y_va, va_prob)

# -------------- Optuna -------------
def _optuna_joint_tune(
    texts_tr: List[str],
    y_tr: List[int],
    args,
) -> Tuple[Dict, Dict]:
    try:
        import optuna
    except ImportError as e:
        raise SystemExit("Optuna is not installed. Please run: pip install optuna") from e
    from sklearn.model_selection import StratifiedKFold

    mf_candidates = [int(x) for x in args.mf_candidates.split(",")]
    ng_candidates = [int(x) for x in args.ng_candidates.split(",")]
    hidden_candidates = [int(x) for x in args.hidden_candidates.split(",")]
    dropout_candidates = [float(x) for x in args.dropout_candidates.split(",")]
    init_candidates = [x.strip() for x in args.init_candidates.split(",")]

    class_weights_global = _compute_class_weights(y_tr) if args.class_weighting else None

    print("\n=== [Stage 2] Joint tuning with Optuna (features + head) on TRAIN only ===")
    print(f"Trials: {args.n_trials} | inner_k: {args.inner_k} | seed: {args.seed}")

    def objective(trial: "optuna.trial.Trial") -> float:
        # ---- Sample trial params (features + head) ----
        max_features = trial.suggest_categorical("max_features", mf_candidates)
        ngram_max   = trial.suggest_categorical("ngram_max", ng_candidates)
        hidden      = trial.suggest_categorical("hidden", hidden_candidates)
        dropout     = trial.suggest_categorical("dropout", dropout_candidates)
        init_scheme = trial.suggest_categorical("init_scheme", init_candidates)
        lr          = trial.suggest_float("lr", args.lr_min, args.lr_max, log=True)
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

            # Vectorize per fold with THIS TRIAL'S feature params (fit on fold-train only)
            vec = TfidfFeaturizer(
                ngram_max=ngram_max,
                max_features=max_features,
                min_df=args.min_df,
                max_df=args.max_df,
            )
            X_tr = vec.fit_transform(x_tr_fold)
            X_va = vec.transform(x_va_fold)

            # Optional: class weights per fold
            cw = _compute_class_weights(y_tr_fold) if args.class_weighting else class_weights_global

            # Train/eval once for this fold
            metrics = _train_eval_once(
                X_tr, y_tr_fold, X_va, y_va_fold,
                args=args,
                hidden=hidden, dropout=dropout, init_scheme=init_scheme,
                lr=lr, weight_decay=weight_decay,
                class_weights=cw, in_dim=X_tr.shape[1],
                desc_prefix=f"[T{trial.number:03d}|F{fold_id}/{args.inner_k}] ",
            )
            f1s.append(metrics["macro_f1"])

            # Report/prune
            trial.report(float(np.mean(f1s)), step=fold_id)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_f1 = float(np.mean(f1s))
        return mean_f1

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

    # ---- Joint Optuna (features + head) on TRAIN ONLY ----
    best_params, best_info = _optuna_joint_tune(x_tr, y_tr, args)

    # Unpack best
    max_features = int(best_params["max_features"])
    ngram_max    = int(best_params["ngram_max"])
    hidden       = int(best_params["hidden"])
    dropout      = float(best_params["dropout"])
    init_scheme  = best_params["init_scheme"]
    lr           = float(best_params["lr"])
    weight_decay = float(0.0 if best_params.get("wd_choice", None) == "zero" else best_params.get("weight_decay", 0.0))

    # ---- Final vectorizer fit on TRAIN ----
    print("\n=== [Stage 3] Fit final vectorizer on TRAIN and transform VAL/TEST ===")
    feat = TfidfFeaturizer(
        ngram_max=ngram_max, max_features=max_features,
        min_df=args.min_df, max_df=args.max_df,
    )
    Xtr = feat.fit_transform(x_tr)
    Xva = feat.transform(x_va)
    Xte = feat.transform(test_texts)
    os.makedirs(args.out_dir, exist_ok=True)
    feat.save(args.vectorizer_out)
    print(f"... saved vectorizer to {args.vectorizer_out} (vocab size={Xtr.shape[1]})")

    # ---- Final train on TRAIN; eval on VAL ----
    print("\n=== [Stage 4] Train final model on TRAIN; evaluate on VAL ===")
    tr_loader = make_loader(Xtr, y_tr, batch_size=args.batch_size, shuffle=True)
    va_loader = make_loader(Xva, y_va, batch_size=args.batch_size, shuffle=False)
    te_loader = make_loader(Xte, None, batch_size=args.batch_size, shuffle=False)

    cw = _compute_class_weights(y_tr) if args.class_weighting else None
    trainer = TrainerTFIDF(
        in_dim=Xtr.shape[1],
        hidden=hidden, dropout=dropout,
        use_layernorm=args.use_layernorm,
        init_scheme=init_scheme,
        lr=lr, weight_decay=weight_decay,
        max_epochs=args.epochs, patience=args.patience,
        class_weights=cw,
        desc_prefix="[FINAL] ",
    )
    fit_info = trainer.fit(tr_loader, va_loader)
    best_epoch = int(fit_info.get("best_epoch", args.epochs))
    print(f"[FINAL] Best epoch on VAL: {best_epoch}")

    print("\n=== [Stage 5] Validation metrics & plots ===")
    val_prob = trainer.predict_proba(va_loader)
    metrics = evaluate_and_plot(y_va, val_prob, out_dir=args.out_dir, prefix="val")
    print(json.dumps(metrics, indent=2))

    # ---- Optional: retrain on TRAIN+VAL for submission (run exactly best_epoch epochs) ----
    if args.retrain_full or args.make_submission:
        print(f"\n=== [Stage 6] Retrain on TRAIN+VAL for {best_epoch} epoch(s) & prepare submission ===")
        x_full = x_tr + x_va
        y_full = y_tr + y_va
        feat_full = TfidfFeaturizer(
            ngram_max=ngram_max, max_features=max_features,
            min_df=args.min_df, max_df=args.max_df,
        )
        Xfull = feat_full.fit_transform(x_full)
        Xtest = feat_full.transform(test_texts)
        feat_full.save(args.vectorizer_out.replace(".pkl", "_full.pkl"))
        print(f"... saved full vectorizer to {args.vectorizer_out.replace('.pkl','_full.pkl')} (vocab size={Xfull.shape[1]})")

        cw_full = _compute_class_weights(y_full) if args.class_weighting else None
        tr_full = make_loader(Xfull, y_full, batch_size=args.batch_size, shuffle=True)
        te_full = make_loader(Xtest, None, batch_size=args.batch_size, shuffle=False)

        trainer_full = TrainerTFIDF(
            in_dim=Xfull.shape[1],
            hidden=hidden, dropout=dropout,
            use_layernorm=args.use_layernorm,
            init_scheme=init_scheme,
            lr=lr, weight_decay=weight_decay,
            max_epochs=best_epoch,          # match Stage 4 best epoch
            patience=args.patience,
            class_weights=cw_full,
            desc_prefix="[FULL] ",
        )
        trainer_full.fit(tr_full, None)

        if args.make_submission:
            print("\n=== [Stage 7] Writing submission ===")
            test_pred = trainer_full.predict(te_full)
            write_submission(test_pred, path=args.submission_out)

if __name__ == "__main__":
    main()
