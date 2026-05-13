# PD Model with Polars + Explainable Boosting Trees (EBT)
> Adapted 10-step pipeline using Polars (data) + EBT (modeling)
> Follows core methodology from PD_MODEL_TEMPLATE.md with algorithm substitutions

---

## CONFIG — fill these in before running

```python
DATA_PATH        = path/to/your_dataset.csv
TARGET_COL       = label          # binary: 1=default/event, 0=non-default/non-event
TEXT_COLS        = []             # free-text columns for NLP features; [] if none
OUTPUT_DIR       = ./             # root output dir
AUTHOR           = Your Name
PROJECT_NAME     = Your_Project_Name
```

---

## Environment Setup

```bash
pip install polars pandas numpy scipy matplotlib seaborn scikit-learn \
            interpret shap bayesian-optimization optbinning
# EBT comes via: from interpret.glassbox import ExplainableBoostingClassifier
```

---

## 10-Step Pipeline (Polars + EBT)

### Step 1 — Data Profiling (Polars)

```python
import polars as pl
import pandas as pd

df_pl = pl.read_csv(DATA_PATH)

# Profile
print(f"Shape: {df_pl.shape}")
print(f"Dtypes:\n{df_pl.dtypes}")
print(f"\nMissing values:\n{df_pl.null_count()}")
print(f"\nTarget distribution:\n{df_pl[TARGET_COL].value_counts()}")

# Convert to pandas for sklearn compatibility (downstream steps)
df = df_pl.to_pandas()
```

### Step 2 — NLP Feature Extraction *(skip if TEXT_COLS is empty)*

```python
from skills.nlp_txn_extractor import TransactionNLPExtractor

extractor = TransactionNLPExtractor()
for col in TEXT_COLS:
    df = extractor.transform(df, col=col)
```

### Step 3 — Train/Test Split

```python
from sklearn.model_selection import train_test_split

X_all = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.25, stratify=y, random_state=42
)

print(f"Train: {X_tr_raw.shape} | Test: {X_te_raw.shape}")
print(f"Train event rate: {y_tr.mean():.4f} | Test: {y_te.mean():.4f}")
```

### Step 4 — optbinning WoE/IV

```python
from optbinning import OptimalBinning

def fit_optbinning(feature_name, x_train, y_train, dtype='numerical'):
    kwargs = dict(name=feature_name, dtype=dtype, solver='cp',
                  max_n_bins=10, min_bin_size=0.03)
    if dtype == 'numerical':
        kwargs['monotonic_trend'] = 'auto'
    ob = OptimalBinning(**kwargs)
    x = x_train.values.astype(float if dtype == 'numerical' else str)
    ob.fit(x, y_train.values)
    bt = ob.binning_table.build()
    iv = float(bt['IV'].iloc[:-2].sum())
    return ob, iv

# Identify feature types
numeric_features = X_tr_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_tr_raw.select_dtypes(include=['object']).columns.tolist()

ob_dict = {}
iv_rows = []
for col in numeric_features + categorical_features:
    dtype = 'numerical' if col in numeric_features else 'categorical'
    x = X_tr_raw[col] if dtype == 'numerical' else X_tr_raw[col].fillna('Missing')
    ob, iv = fit_optbinning(col, x, y_tr, dtype=dtype)
    ob_dict[col] = ob
    iv_rows.append({'Feature': col, 'IV': round(iv, 4)})

# Feature selection
def iv_strength(iv):
    if iv >= 1.0:  return 'Suspected Leakage'
    if iv >= 0.5:  return 'Very Strong'
    if iv >= 0.3:  return 'Strong'
    if iv >= 0.1:  return 'Medium'
    if iv >= 0.02: return 'Weak'
    return 'Useless'

iv_df = pd.DataFrame(iv_rows).sort_values('IV', ascending=False).reset_index(drop=True)
iv_df['Strength'] = iv_df['IV'].apply(iv_strength)
iv_df.to_csv(f'{OUTPUT_DIR}/data/iv_table.csv', index=False)

selected_iv = iv_df[(iv_df['IV'] < 1.0) & (iv_df['IV'] >= 0.02)]['Feature'].tolist()
print(f"Selected {len(selected_iv)} features (IV range: {iv_df['IV'].min():.4f} - {iv_df['IV'].max():.4f})")
```

Apply WoE encoding:

```python
def apply_woe(X_df, ob_dict, features):
    out = pd.DataFrame(index=X_df.index)
    for col in features:
        ob = ob_dict[col]
        x = X_df[col].values.astype(float) if ob.dtype == 'numerical' \
            else X_df[col].fillna('Missing').values.astype(str)
        out[col] = ob.transform(x, metric='woe')
    return out.fillna(0.0)

X_tr_woe = apply_woe(X_tr_raw, ob_dict, selected_iv)
X_te_woe = apply_woe(X_te_raw, ob_dict, selected_iv)
```

### Step 5 — WoE PDP Plots

```python
import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots((len(numeric_features) + 1) // 2, 2, figsize=(14, 4 * ((len(numeric_features) + 1) // 2)))
axes = axes.flatten()

for idx, feat in enumerate(numeric_features):
    if feat not in ob_dict:
        continue
    ob = ob_dict[feat]
    bt = ob.binning_table.build().iloc[:-2]
    ax = axes[idx]
    ax2 = ax.twinx()
    ax.bar(range(len(bt)), bt['Count (%)'], color='#5b9bd5', alpha=0.55)
    ax2.plot(range(len(bt)), bt['Event rate'], color='#c00000', marker='o', linewidth=2)
    ax.set_title(feat, fontweight='bold')
    ax.set_ylabel('Count (%)')
    ax2.set_ylabel('Event Rate')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_woe_pdp.png', bbox_inches='tight', dpi=120)
plt.close()
```

### Step 6 — SHAP RFE (Using Polars-compatible features)

```python
from probatus.feature_elimination import ShapRFECV
import lightgbm as lgb

# Use LightGBM for RFE (for feature selection only; EBT used for final model)
lgbm_rfe = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    num_leaves=20, min_child_samples=30, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbose=-1,
)

shap_rfe = ShapRFECV(lgbm_rfe, step=0.2, cv=5, scoring='roc_auc',
                      n_jobs=1, random_state=42)

results_df = shap_rfe.fit_compute(X_tr_woe, y_tr)
best_n = results_df.loc[results_df['val_metric_mean'].idxmax(), 'num_features']
selected_features = shap_rfe.get_reduced_features_set(num_features=int(best_n))

print(f"Selected {len(selected_features)} features via SHAP RFE")

# Plot elbow curve
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(results_df['num_features'], results_df['val_metric_mean'],
            yerr=results_df.get('val_metric_std', 0), fmt='o-', color='#1f4e79', capsize=3)
ax.axvline(best_n, color='#c00000', linestyle='--', label=f'Selected: {best_n} features')
ax.set_xlabel('Number of Features')
ax.set_ylabel('CV AUC')
ax.set_title('SHAP RFE Elbow Curve')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_rfe.png', bbox_inches='tight', dpi=120)
plt.close()
```

### Step 7 — Model Training with EBT

```python
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.metrics import roc_auc_score

# Train EBT
ebt_model = ExplainableBoostingClassifier(
    interactions=10,
    outer_bags=8,
    inner_bags=4,
    learning_rate=0.05,
    max_rounds=5000,
    random_state=42,
    n_jobs=-1
)

ebt_model.fit(X_tr_woe[selected_features], y_tr)

# Predictions
tr_proba = ebt_model.predict_proba(X_tr_woe[selected_features])[:, 1]
te_proba = ebt_model.predict_proba(X_te_woe[selected_features])[:, 1]

tr_auc = roc_auc_score(y_tr, tr_proba)
te_auc = roc_auc_score(y_te, te_proba)

print(f"Train AUC: {tr_auc:.4f} | Test AUC: {te_auc:.4f} | Gap: {(tr_auc - te_auc):.4f}")
```

### Step 8 — Evaluation (7 Figures)

```python
from skills.model_utils import plot_auc2, plot_confusion, exp_vs_act, plot_decile_chart
from sklearn.metrics import confusion_matrix
import shap

# Figure 1: ROC + KS
plot_auc2(y_tr, tr_proba, y_te, te_proba, title=PROJECT_NAME,
          save_path=f'{OUTPUT_DIR}/figures/fig_roc_ks.png')

# Figure 2: Confusion Matrix
plot_confusion(y_te, (te_proba >= 0.5).astype(int),
               save_path=f'{OUTPUT_DIR}/figures/fig_confusion.png')

# Figure 3 & 4: SHAP (using EBT)
explainer = shap.TreeExplainer(ebt_model.model_)
shap_vals = explainer.shap_values(X_te_woe[selected_features])
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

mean_shap = np.abs(shap_vals).mean(axis=0)
shap_df = pd.DataFrame({'feature': selected_features, 'importance': mean_shap}).sort_values('importance')

fig, ax = plt.subplots(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
ax.barh(shap_df['feature'], shap_df['importance'], color='#1f4e79')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance (EBT, Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_bar.png', bbox_inches='tight', dpi=120)
plt.close()

fig = plt.figure(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
shap.summary_plot(shap_vals, X_te_woe[selected_features], show=False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_summary.png', bbox_inches='tight', dpi=120)
plt.close()

# Figure 5: Decile Chart
tbl = exp_vs_act(y_te, te_proba)
plot_decile_chart(tbl, save_path=f'{OUTPUT_DIR}/figures/fig_decile.png')
tbl.to_csv(f'{OUTPUT_DIR}/data/table_decile.csv', index=False)

# Figure 6: Score Distribution
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(te_proba[y_te == 0], bins=40, density=True, alpha=0.6,
        color='#5b9bd5', label='Non-Default (y=0)')
ax.hist(te_proba[y_te == 1], bins=40, density=True, alpha=0.6,
        color='#c00000', label='Default (y=1)')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution — Test Set', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_score_dist.png', bbox_inches='tight', dpi=120)
plt.close()

# Figure 7: Correlation Heatmap
import seaborn as sns
corr = X_tr_woe[selected_features].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=np.triu(np.ones_like(corr, dtype=bool), k=1),
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix (WoE, Train)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_corr.png', bbox_inches='tight', dpi=120)
plt.close()
```

### Step 9 — Persist Outputs

```python
import pickle

bundle = {
    'model':            ebt_model,
    'features':         selected_features,
    'ob_dict':          ob_dict,
    'X_tr':             X_tr_woe[selected_features],
    'X_te':             X_te_woe[selected_features],
    'y_tr':             y_tr,
    'y_te':             y_te,
    'tr_auc':           tr_auc,
    'te_auc':           te_auc,
    'tr_proba':         tr_proba,
    'te_proba':         te_proba,
}

with open(f'{OUTPUT_DIR}/models/final_model_ebt.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print(f"✓ Model saved to {OUTPUT_DIR}/models/final_model_ebt.pkl")
```

### Step 10 — Print Summary

```python
print("=" * 50)
print(f"PROJECT:  {PROJECT_NAME}")
print(f"AUTHOR:   {AUTHOR}")
print("=" * 50)
print(f"Dataset:  {len(df)} rows | {y.sum()} events ({100*y.mean():.1f}%)")
print(f"Features: {len(selected_features)} selected")
print(f"Model:    Explainable Boosting Trees (EBT)")
print("-" * 50)
print(f"          AUC")
print(f"Train:    {tr_auc:.4f}")
print(f"Test:     {te_auc:.4f}")
print(f"Gap:      {(tr_auc - te_auc):.4f} {'✓ stable' if (tr_auc - te_auc) < 0.02 else '⚠ overfitting'}")
print("-" * 50)
top_decile_pct = 100 * y_te[te_proba >= te_proba.quantile(0.9)].sum() / y_te.sum()
print(f"Top decile captures {top_decile_pct:.1f}% of test events")
print("=" * 50)
```

---

## Key Differences from LightGBM Template

| Aspect | LightGBM | EBT |
|--------|----------|-----|
| **Model** | `LGBMClassifier` | `ExplainableBoostingClassifier` |
| **Interpretability** | SHAP required | Built-in global explanations |
| **Hyperparameters** | Learning rate, num_leaves, depth | interactions, bags, learning_rate |
| **Optimization** | Bayesian optimization | Can use fixed params (pre-tuned) |
| **Overfitting** | Monitor via early_stopping | Monitor via train/test AUC gap |

---

## Full Pipeline Script

See `run_pipeline_polars_ebt.py` for the complete end-to-end implementation.

