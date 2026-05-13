"""
woe_iv.py — Weight of Evidence / Information Value
===================================================
Pure pandas/numpy. No optbinning or external credit libraries required.
Works on Python 3.13+.

Usage
-----
    from skills.woe_iv import compute_all_iv, plot_woe_pdp, apply_woe

    iv_df, bins = compute_all_iv(df, target_col='label',
                                 numeric_cols=num_feats,
                                 categorical_cols=cat_feats)
    df_woe = apply_woe(df, bins)
    plot_woe_pdp('feature_7', bins['feature_7'])
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# ── IV strength labels ──────────────────────────────────────────────────────

def iv_strength(iv: float) -> str:
    if iv < 0.02:  return "Useless"
    if iv < 0.10:  return "Weak"
    if iv < 0.30:  return "Medium"
    if iv < 0.50:  return "Strong"
    if iv < 1.00:  return "Very Strong"
    return "Suspicious (Leakage?)"


# ── Core WoE/IV calculators ─────────────────────────────────────────────────

def _woe_stats(df_tmp: pd.DataFrame, total_events: int, total_nonevents: int,
               eps: float = 0.5) -> pd.DataFrame:
    """Add pct_events, pct_nonevents, woe, iv_bin columns to an aggregated df."""
    df_tmp = df_tmp.copy()
    df_tmp["pct_events"]    = (df_tmp["events"]    + eps) / total_events
    df_tmp["pct_nonevents"] = (df_tmp["nonevents"] + eps) / total_nonevents
    df_tmp["event_rate"]    = df_tmp["events"] / df_tmp["count"]
    df_tmp["nonevent_rate"] = df_tmp["nonevents"] / df_tmp["count"]
    df_tmp["woe"]           = np.log(df_tmp["pct_events"] / df_tmp["pct_nonevents"])
    df_tmp["iv_bin"]        = (df_tmp["pct_events"] - df_tmp["pct_nonevents"]) * df_tmp["woe"]
    return df_tmp


def compute_woe_iv_numeric(series: pd.Series, target: pd.Series,
                            n_bins: int = 10) -> tuple[pd.DataFrame, float]:
    """
    WoE/IV for a numeric feature using quantile binning.
    Binary/low-cardinality features are automatically treated as categorical.

    Returns (bins_df, iv_total).
    """
    df_tmp = pd.DataFrame({"x": series, "y": target})
    total_events    = (df_tmp["y"] == 1).sum()
    total_nonevents = (df_tmp["y"] == 0).sum()

    non_null = df_tmp.dropna(subset=["x"])
    if non_null["x"].nunique() <= 2:
        return compute_woe_iv_categorical(series, target)

    try:
        non_null = non_null.copy()
        non_null["bin"] = pd.qcut(non_null["x"], q=n_bins, duplicates="drop")
    except ValueError:
        non_null = non_null.copy()
        non_null["bin"] = pd.cut(non_null["x"], bins=n_bins)

    rows = []
    for b, grp in non_null.groupby("bin", observed=True):
        rows.append({"bin": str(b),
                     "events":    (grp["y"] == 1).sum(),
                     "nonevents": (grp["y"] == 0).sum(),
                     "count":     len(grp)})

    miss = df_tmp[df_tmp["x"].isna()]
    if len(miss):
        rows.append({"bin": "Missing",
                     "events":    (miss["y"] == 1).sum(),
                     "nonevents": (miss["y"] == 0).sum(),
                     "count":     len(miss)})

    result = _woe_stats(pd.DataFrame(rows).set_index("bin"),
                        total_events, total_nonevents)
    return result, result["iv_bin"].sum()


def compute_woe_iv_categorical(series: pd.Series, target: pd.Series
                                ) -> tuple[pd.DataFrame, float]:
    """WoE/IV for a categorical or binary feature (one bin per value + Missing)."""
    df_tmp = pd.DataFrame({"x": series.fillna("Missing"), "y": target})
    total_events    = (df_tmp["y"] == 1).sum()
    total_nonevents = (df_tmp["y"] == 0).sum()

    agg = df_tmp.groupby("x").agg(
        events=   ("y", lambda s: (s == 1).sum()),
        nonevents=("y", lambda s: (s == 0).sum()),
        count=    ("y", "count"),
    )
    result = _woe_stats(agg, total_events, total_nonevents)
    return result, result["iv_bin"].sum()


# ── Batch IV computation ────────────────────────────────────────────────────

def compute_all_iv(df: pd.DataFrame,
                   target_col: str,
                   numeric_cols: list[str],
                   categorical_cols: list[str],
                   n_bins: int = 10) -> tuple[pd.DataFrame, dict]:
    """
    Compute IV for all features. Returns:
      iv_df  — DataFrame with columns [feature, IV, Strength], sorted desc
      bins   — dict mapping feature name → bins_df (for WoE encoding / PDPs)
    """
    y = df[target_col]
    iv_records, bins = [], {}

    for feat in numeric_cols:
        try:
            b, iv = compute_woe_iv_numeric(df[feat], y, n_bins)
            iv_records.append({"feature": feat, "IV": round(iv, 4),
                                "Strength": iv_strength(iv)})
            bins[feat] = b
        except Exception as e:
            print(f"  Skipping {feat}: {e}")

    for feat in categorical_cols:
        b, iv = compute_woe_iv_categorical(df[feat], y)
        iv_records.append({"feature": feat, "IV": round(iv, 4),
                            "Strength": iv_strength(iv)})
        bins[feat] = b

    iv_df = (pd.DataFrame(iv_records)
             .sort_values("IV", ascending=False)
             .reset_index(drop=True))
    return iv_df, bins


# ── WoE encoding ────────────────────────────────────────────────────────────

def apply_woe(df: pd.DataFrame, bins: dict,
              suffix: str = "_woe") -> pd.DataFrame:
    """
    Replace each feature in bins with its WoE-encoded value.
    Adds new columns named <feature><suffix>; originals are kept.
    """
    df = df.copy()
    for feat, b in bins.items():
        woe_map = b["woe"].to_dict()
        if feat in df.columns:
            df[feat + suffix] = df[feat].astype(str).map(woe_map).fillna(0)
    return df


# ── WoE Partial Dependence Plot ─────────────────────────────────────────────

def plot_woe_pdp(feature_name: str, bins_df: pd.DataFrame,
                 ax=None, figsize=(10, 4)) -> None:
    """
    Three-layer WoE PDP:
      • Bar chart  — observation count per bin
      • Line (red) — WoE per bin
      • Line (green) — default rate % per bin
    """
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    b = bins_df.reset_index()
    x = np.arange(len(b))
    labels = [str(v)[:22] for v in b.iloc[:, 0]]

    ax.bar(x, b["count"], color="#90CAF9", alpha=0.8)
    ax.set_ylabel("Count", color="#1565C0")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_title(feature_name)

    ax2 = ax.twinx()
    ax2.plot(x, b["woe"], "o-", color="#D32F2F", lw=2, ms=5, label="WoE")
    ax2.axhline(0, color="grey", lw=0.8, linestyle="--")
    ax2.set_ylabel("WoE", color="#D32F2F")

    ax3 = ax.twinx()
    ax3.spines["right"].set_position(("outward", 55))
    ax3.plot(x, b["event_rate"] * 100, "s--", color="#388E3C",
             lw=1.5, ms=4, label="Default Rate %")
    ax3.set_ylabel("Default Rate %", color="#388E3C")
    ax3.yaxis.set_major_formatter(mtick.PercentFormatter())


def plot_all_woe_pdps(bins: dict, features: list[str],
                      cols: int = 2, figsize_per=(9, 4)) -> None:
    """Plot WoE PDPs for a list of features in a grid layout."""
    rows = (len(features) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols,
                              figsize=(figsize_per[0] * cols,
                                       figsize_per[1] * rows))
    axes = np.array(axes).flatten()
    for i, feat in enumerate(features):
        if feat in bins:
            plot_woe_pdp(feat, bins[feat], ax=axes[i])
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.tight_layout()
    plt.show()
