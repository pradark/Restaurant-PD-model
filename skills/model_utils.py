"""
model_utils.py — Model evaluation utilities for binary classification
=====================================================================
Reusable functions for credit / PD model evaluation.

Usage
-----
    from skills.model_utils import (
        check_missing, performance_metrics,
        plot_auc2, exp_vs_act, plot_decile_chart
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report


# ── Data quality ────────────────────────────────────────────────────────────

def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary of columns with missing values, sorted by % missing."""
    miss     = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    result   = pd.DataFrame({"Missing": miss, "Missing_%": miss_pct})
    return result[result["Missing"] > 0].sort_values("Missing_%", ascending=False)


# ── Core metrics ────────────────────────────────────────────────────────────

def performance_metrics(y_true: np.ndarray, y_proba: np.ndarray,
                         label: str = "") -> tuple[float, float]:
    """Print and return (AUC, KS) for a set of predictions."""
    auc = roc_auc_score(y_true, y_proba)
    ks  = ks_2samp(y_proba[y_true == 1], y_proba[y_true == 0]).statistic
    print(f"{label:15s}  AUC = {auc:.4f}   KS = {ks:.4f}")
    return auc, ks


def gap_check(tr_auc: float, te_auc: float,
              tr_ks: float,  te_ks: float,
              threshold: float = 0.02) -> None:
    """Print train/test gap and flag if it exceeds threshold."""
    auc_gap = tr_auc - te_auc
    ks_gap  = tr_ks  - te_ks
    ok = lambda g: "✓" if g < threshold else "✗ EXCEEDS TARGET"
    print(f"AUC gap : {auc_gap*100:.2f}%  {ok(auc_gap)}")
    print(f"KS  gap : {ks_gap*100:.2f}%   {ok(ks_gap)}")


# ── ROC + KS plot ───────────────────────────────────────────────────────────

def plot_auc2(y_tr, tr_proba, y_te, te_proba, title: str = "",
              save_path: str = None) -> None:
    """
    Side-by-side ROC curve and KS separation plot for train and test.
    Optionally saves to save_path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ROC
    ax = axes[0]
    for y, p, lbl, c in [(y_tr, tr_proba, "Train", "#1f77b4"),
                          (y_te, te_proba, "Test",  "#ff7f0e")]:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=c, lw=2, label=f"{lbl} (AUC={auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title} — ROC Curve")
    ax.legend()

    # KS
    ax = axes[1]
    for y, p, lbl, c in [(y_tr, tr_proba, "Train", "#1f77b4"),
                          (y_te, te_proba, "Test",  "#ff7f0e")]:
        df_ks = (pd.DataFrame({"y": y, "p": p})
                 .sort_values("p", ascending=False)
                 .reset_index(drop=True))
        n_ev  = (df_ks["y"] == 1).sum()
        n_nev = (df_ks["y"] == 0).sum()
        df_ks["cum_ev"]  = (df_ks["y"] == 1).cumsum() / n_ev
        df_ks["cum_nev"] = (df_ks["y"] == 0).cumsum() / n_nev
        ks_val = ks_2samp(p[y == 1], p[y == 0]).statistic
        pct = np.linspace(0, 1, len(df_ks))
        ax.plot(pct, df_ks["cum_ev"],  color=c, lw=2, linestyle="-",
                label=f"{lbl} Events")
        ax.plot(pct, df_ks["cum_nev"], color=c, lw=2, linestyle="--",
                label=f"{lbl} Non-events (KS={ks_val:.4f})")
    ax.set_xlabel("Population %")
    ax.set_ylabel("Cumulative %")
    ax.set_title(f"{title} — KS Statistic")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()


# ── Confusion matrix ────────────────────────────────────────────────────────

def plot_confusion(y_true, y_pred, labels=("Non-Default", "Default"),
                   title: str = "Confusion Matrix", save_path: str = None) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=[f"Pred: {l}" for l in labels],
                yticklabels=[f"Actual: {l}" for l in labels])
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()
    print(classification_report(y_true, y_pred, target_names=list(labels)))


# ── Decile expected vs actual ───────────────────────────────────────────────

def exp_vs_act(y_true: np.ndarray, y_proba: np.ndarray,
               n_deciles: int = 10) -> pd.DataFrame:
    """
    Decile-level expected vs actual default rate table.
    Decile 1 = highest predicted risk.
    """
    df = (pd.DataFrame({"y": y_true, "p": y_proba})
          .sort_values("p", ascending=False)
          .reset_index(drop=True))
    df["decile"] = pd.qcut(df.index, n_deciles, labels=range(1, n_deciles + 1))
    tbl = df.groupby("decile", observed=False).agg(
        n=              ("y", "count"),
        actual_defaults=("y", "sum"),
        expected_rate=  ("p", "mean"),
    ).reset_index()
    tbl["actual_rate"]      = tbl["actual_defaults"] / tbl["n"]
    tbl["actual_rate_pct"]  = (tbl["actual_rate"]   * 100).round(1)
    tbl["expected_rate_pct"]= (tbl["expected_rate"] * 100).round(1)
    return tbl


def plot_decile_chart(tbl: pd.DataFrame, title: str = "Expected vs Actual Default Rate",
                      save_path: str = None) -> None:
    """Bar chart of actual vs expected default rate by decile."""
    fig, ax = plt.subplots(figsize=(11, 6))
    x, w = np.arange(len(tbl)), 0.35
    b1 = ax.bar(x - w/2, tbl["actual_rate_pct"],   w, label="Actual %",   color="#1f77b4", alpha=0.85)
    b2 = ax.bar(x + w/2, tbl["expected_rate_pct"], w, label="Expected %", color="#ff7f0e", alpha=0.85)
    for b in b1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{b.get_height():.1f}%", ha="center", fontsize=8)
    for b in b2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3,
                f"{b.get_height():.1f}%", ha="center", fontsize=8, color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"D{i}" for i in tbl["decile"]])
    ax.set_xlabel("Decile (1 = highest risk)")
    ax.set_ylabel("Default Rate (%)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.show()
