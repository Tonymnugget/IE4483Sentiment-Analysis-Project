# eval_gemma.py
from __future__ import annotations
from typing import Dict, List
import os, json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, roc_auc_score,
    average_precision_score, roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.calibration import calibration_curve
import pandas as pd

def compute_metrics(y_true: List[int], y_prob: np.ndarray) -> Dict:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = y_prob.argmax(axis=-1)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=[0, 1])
    pos = y_prob[:, 1]
    try:
        roc_auc = roc_auc_score(y_true, pos)
    except ValueError:
        roc_auc = float("nan")
    try:
        pr_auc = average_precision_score(y_true, pos)
    except ValueError:
        pr_auc = float("nan")
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "per_class": {
            "class_0": {"precision": float(pr[0]), "recall": float(rc[0]), "f1": float(f1[0])},
            "class_1": {"precision": float(pr[1]), "recall": float(rc[1]), "f1": float(f1[1])},
        },
    }

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _plot_curve(x, y, xlabel, ylabel, title, out_path, label=None):
    plt.figure()
    if label:
        plt.plot(x, y, lw=2, label=label)
        plt.legend()
    else:
        plt.plot(x, y, lw=2)
    plt.xlabel(xlabel); plt.ylabel(ylabel); plt.title(title)
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_roc(y_true, y_prob1, out_path, auc_val: float | None = None):
    fpr, tpr, _ = roc_curve(y_true, y_prob1)
    label = f"ROC (AUC = {auc_val:.4f})" if auc_val is not None and np.isfinite(auc_val) else "ROC"
    plt.figure()
    plt.plot(fpr, tpr, lw=2, label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(loc="lower right"); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_pr(y_true, y_prob1, out_path, ap_val: float | None = None):
    precision, recall, _ = precision_recall_curve(y_true, y_prob1)
    pos_rate = float(np.mean(y_true))
    label = f"PR (AP = {ap_val:.4f})" if ap_val is not None and np.isfinite(ap_val) else "PR"
    plt.figure()
    plt.plot(recall, precision, lw=2, label=label)
    plt.hlines(pos_rate, 0, 1, linestyles="--", label=f"Baseline = {pos_rate:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve")
    plt.legend(loc="lower left"); plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_confusion(y_true, y_pred, out_path, normalize=False):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1], normalize=("true" if normalize else None))
    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", aspect="auto")
    plt.colorbar(im)
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ["0", "1"]); plt.yticks(tick_marks, ["0", "1"])
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            plt.text(j, i, format(val, fmt),
                     ha="center", va="center",
                     color="white" if val > thresh else "black")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.title("Confusion Matrix" + (" (Normalized)" if normalize else ""))
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_calibration(y_true, y_prob1, out_path, n_bins=15):
    prob_true, prob_pred = calibration_curve(y_true, y_prob1, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(prob_pred, prob_true, marker="o")
    plt.xlabel("Mean predicted probability"); plt.ylabel("Fraction of positives")
    plt.title("Calibration (Reliability) Curve")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def plot_score_hist(y_prob1, out_path):
    plt.figure()
    plt.hist(y_prob1, bins=30, alpha=0.9)
    plt.xlabel("P(class=1)"); plt.ylabel("Count"); plt.title("Score Histogram")
    plt.tight_layout(); plt.savefig(out_path, dpi=150); plt.close()

def evaluate_and_plot(y_true: List[int], y_prob: np.ndarray, out_dir: str, prefix: str = "val") -> Dict:
    _ensure_dir(out_dir)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = y_prob.argmax(axis=-1)
    pos = y_prob[:, 1]

    metrics = compute_metrics(y_true, y_prob)
    with open(os.path.join(out_dir, f"{prefix}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Annotated plots
    try: plot_roc(y_true, pos, os.path.join(out_dir, f"{prefix}_roc.png"), auc_val=metrics.get("roc_auc"))
    except ValueError: pass
    try: plot_pr(y_true, pos, os.path.join(out_dir, f"{prefix}_pr.png"), ap_val=metrics.get("pr_auc"))
    except ValueError: pass
    plot_confusion(y_true, y_pred, os.path.join(out_dir, f"{prefix}_confusion.png"), normalize=False)
    plot_confusion(y_true, y_pred, os.path.join(out_dir, f"{prefix}_confusion_norm.png"), normalize=True)
    try: plot_calibration(y_true, pos, os.path.join(out_dir, f"{prefix}_calibration.png"))
    except Exception: pass
    plot_score_hist(pos, os.path.join(out_dir, f"{prefix}_score_hist.png"))
    return metrics

def write_submission(pred_labels: np.ndarray, path: str = "submission.csv"):
    df = pd.DataFrame({"id": np.arange(len(pred_labels)), "label": pred_labels.astype(int)})
    df.to_csv(path, index=False)
    print(f"Wrote submission to {path}")
