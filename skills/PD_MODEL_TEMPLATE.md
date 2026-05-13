# PD Model Agent Template
> Copy this file and the `skills/` folder to any new project.
> Fill in the CONFIG block, then feed this file to an agent as the opening prompt.

---

## CONFIG — fill these in before handing to an agent

```
DATA_PATH        = path/to/your_dataset.csv
TARGET_COL       = label          # binary: 1=default/event, 0=non-default/non-event
TEXT_COLS        = []             # free-text columns to extract NLP features from; [] if none
OUTPUT_DIR       = ./             # root output dir; sub-dirs created automatically
AUTHOR           = Your Name
PROJECT_NAME     = Your_Project_Name
PYTHON_ENV       = optbinning_env # conda env name with Python 3.11 + optbinning
```

---

## Environment Setup

optbinning requires Python 3.11 (ortools conflict with Python 3.13 / ARM64).
Run all steps in a dedicated conda environment:

```bash
conda create -n optbinning_env python=3.11 -y
conda activate optbinning_env
pip install optbinning==0.21.0 lightgbm probatus shap bayesian-optimization \
            pandas numpy scipy matplotlib seaborn scikit-learn
# Fix libomp on ARM Mac (copy from base anaconda):
cp /opt/anaconda3/lib/libomp.dylib /opt/anaconda3/envs/optbinning_env/lib/libomp.dylib
```

If optbinning is not needed (no monotonic WoE constraint), the full pipeline
runs in any Python 3.x environment using `skills/woe_iv.py` instead.

---

## Agent Prompt

You are a credit risk data scientist. Build an end-to-end binary probability of default (PD)
model using the dataset and config above. Follow the steps below in order.
Use the skill files in `skills/` for core logic — do not reimplement what is already there.
Run all Python code using the `PYTHON_ENV` conda environment unless stated otherwise.

Create the following output directories before starting:

```python
import os
for d in ['data', 'figures', 'models', 'notebooks', 'paper', 'scripts']:
    os.makedirs(f'{OUTPUT_DIR}/{d}', exist_ok=True)
```

---

### Step 1 — Data Profiling

- Load `DATA_PATH`. Print: shape, dtypes, missing value counts per column, target distribution.
- Identify and replace sentinel values (e.g. values ≥ 9,999,990 → NaN; "N/A", "unknown", "n" → NaN).
- Fix mixed-type columns (e.g. string "0"/"1" mixed with numeric → cast to numeric).
- Print the class imbalance ratio (non-events : events).
- Flag any column with >20% missing values.
- Classify each column as: Identifier, Numeric, Categorical, Binary, Text, or Date.

```python
df['sentinel_col'] = df['sentinel_col'].where(df['sentinel_col'] < 9999990, np.nan)
df['mixed_col']    = pd.to_numeric(df['mixed_col'].replace({'n': np.nan}), errors='coerce')
```

---

### Step 2 — NLP Feature Extraction *(skip if TEXT_COLS is empty)*

For each column in `TEXT_COLS`, extract structured features using `TransactionNLPExtractor`
from `skills/nlp_txn_extractor.py`:

```python
from skills.nlp_txn_extractor import TransactionNLPExtractor

extractor = TransactionNLPExtractor()
df = extractor.transform(df, col=text_col)
```

This adds 7 columns: `merchant_category`, `txn_channel`, `txn_direction`,
`is_recurring`, `is_p2p`, `is_international`, `merchant_risk_tier`.

Extend default patterns via `extra_merchants={}` to match the domain.
All extracted features flow into subsequent steps as regular columns.

---

### Step 3 — Train / Test Split

```python
from sklearn.model_selection import train_test_split

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.25, stratify=y, random_state=42
)
```

- Confirm the event rate is preserved in both splits (within 0.5 pp).
- If a time column exists, use chronological split instead (earlier → train, later → test).
- **Fit all transformers on training data only — never on test.**

---

### Step 4 — optbinning WoE / IV

Use `OptimalBinning` from the `optbinning` package with `monotonic_trend='auto'` for all
numeric features. This automatically detects ascending / descending / peak / valley per feature.
Do not pass `monotonic_trend` for categorical features.

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
    bt  = ob.binning_table.build()
    iv  = float(bt['IV'].iloc[:-2].sum())   # exclude totals rows
    return ob, iv

ob_dict = {}
iv_rows = []
for col in numeric_features + categorical_features:
    dtype = 'numerical' if col in numeric_features else 'categorical'
    x     = X_tr_raw[col] if dtype == 'numerical' else X_tr_raw[col].fillna('Missing')
    ob, iv = fit_optbinning(col, x, y_tr, dtype=dtype)
    ob_dict[col] = ob
    iv_rows.append({'Feature': col, 'IV': round(iv, 4)})
```

IV strength labels and feature selection:

```python
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

# Feature selection: exclude leakage (IV >= 1.0) and useless (IV < 0.02)
selected_iv = iv_df[(iv_df['IV'] < 1.0) & (iv_df['IV'] >= 0.02)]['Feature'].tolist()
```

Apply WoE encoding to train and test using the fitted objects:

```python
def apply_woe(X_df, ob_dict, features):
    out = pd.DataFrame(index=X_df.index)
    for col in features:
        ob   = ob_dict[col]
        x    = X_df[col].values.astype(float) if ob.dtype == 'numerical' \
               else X_df[col].fillna('Missing').values.astype(str)
        out[col] = ob.transform(x, metric='woe')
    return out.fillna(0.0)

X_tr_woe = apply_woe(X_tr_raw, ob_dict, selected_iv)
X_te_woe = apply_woe(X_te_raw, ob_dict, selected_iv)
```

---

### Step 5 — WoE PDP Plots

Plot event rate and bin % for each surviving numeric feature using optbinning's binning table.
Save to `{OUTPUT_DIR}/figures/fig_woe_pdp.png`.

```python
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
for feat in numeric_features_selected:
    ob   = ob_dict[feat]
    bt   = ob.binning_table.build().iloc[:-2]   # drop totals
    ax2  = ax.twinx()
    ax.bar(range(len(bt)), bt['Count (%)'], color='#5b9bd5', alpha=0.55)
    ax2.plot(range(len(bt)), bt['Event rate'], color='#c00000',
             marker='o', linewidth=2)
    ax.set_title(feat, fontweight='bold')
plt.savefig(f'{OUTPUT_DIR}/figures/fig_woe_pdp.png', bbox_inches='tight', dpi=120)
```

---

### Step 6 — SHAP Recursive Feature Elimination

```python
from probatus.feature_elimination import ShapRFECV
import lightgbm as lgb

lgbm_rfe = lgb.LGBMClassifier(
    n_estimators=200, learning_rate=0.05, max_depth=4,
    num_leaves=20, min_child_samples=30, subsample=0.8,
    colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=0.5,
    random_state=42, n_jobs=-1, verbose=-1,
)
shap_rfe = ShapRFECV(lgbm_rfe, step=0.2, cv=5, scoring='roc_auc',
                      n_jobs=1, random_state=42)

results_df       = shap_rfe.fit_compute(X_tr_woe, y_tr)
best_n           = results_df.loc[results_df['val_metric_mean'].idxmax(), 'num_features']
selected_features = shap_rfe.get_reduced_features_set(num_features=int(best_n))
```

- Select at the elbow: max CV AUC. Prefer fewer features when difference < 0.002.
- Print final feature list and CV AUC at selected n.
- Plot elbow curve and save to `{OUTPUT_DIR}/figures/fig_shap_rfe.png`.

```python
fig, ax = plt.subplots(figsize=(9, 5))
ax.errorbar(results_df['num_features'], results_df['val_metric_mean'],
            yerr=results_df.get('val_metric_std', 0), fmt='o-', color='#1f4e79', capsize=3)
ax.axvline(best_n, color='#c00000', linestyle='--', label=f'Selected: {best_n} features')
ax.set_xlabel('Number of Features'); ax.set_ylabel('CV AUC')
ax.legend(); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_rfe.png', bbox_inches='tight', dpi=120)
```

---

### Step 7 — Model Training with Bayesian Optimisation

Use `BayesLGBM` from `skills/bayes_lgbm.py`:

```python
from skills.bayes_lgbm import BayesLGBM

model = BayesLGBM(gap_threshold=0.02, gap_penalty=5.0, n_init=10, n_iter=40, cv=5)
model.fit(X_tr_woe[selected_features], y_tr)
print(model.best_params_)
```

The objective penalises train/val AUC gaps above 2 pp:
`score = val_AUC − 5 × max(0, gap − 0.02)`

Alternatively, call the optimiser directly (as in `run_pipeline_optbinning.py`) if
`skills/bayes_lgbm.py` is not available:

```python
from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(f=objective_fn, pbounds=PBOUNDS, random_state=42, verbose=0)
optimizer.maximize(init_points=10, n_iter=40)
```

---

### Step 8 — Evaluation (7 Figures)

```python
from skills.model_utils import performance_metrics, gap_check, plot_auc2, \
                                plot_confusion, exp_vs_act, plot_decile_chart

tr_proba = model.predict_proba(X_tr_woe[selected_features])[:, 1]
te_proba = model.predict_proba(X_te_woe[selected_features])[:, 1]

tr_auc, tr_ks = performance_metrics(y_tr, tr_proba, label='Train')
te_auc, te_ks = performance_metrics(y_te, te_proba, label='Test')
gap_check(tr_auc, te_auc, tr_ks, te_ks)
```

Generate all seven figures:

**Fig 1 — ROC + KS (train vs test overlay)**
```python
plot_auc2(y_tr, tr_proba, y_te, te_proba, title=PROJECT_NAME,
          save_path=f'{OUTPUT_DIR}/figures/fig_roc_ks.png')
```

**Fig 2 — Confusion Matrix**
```python
plot_confusion(y_te, (te_proba >= 0.5).astype(int),
               save_path=f'{OUTPUT_DIR}/figures/fig_confusion.png')
```

**Fig 3 — SHAP Bar Chart (mean |SHAP|)**
```python
import shap
explainer = shap.TreeExplainer(model.model_)
shap_vals = explainer.shap_values(X_te_woe[selected_features])
if isinstance(shap_vals, list): shap_vals = shap_vals[1]

mean_shap = np.abs(shap_vals).mean(axis=0)
shap_df   = pd.DataFrame({'feature': selected_features, 'importance': mean_shap}).sort_values('importance')
fig, ax   = plt.subplots(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
ax.barh(shap_df['feature'], shap_df['importance'], color='#1f4e79')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_bar.png', bbox_inches='tight', dpi=120)
```

**Fig 4 — SHAP Beeswarm (summary plot)**
```python
fig = plt.figure(figsize=(8, max(4, len(selected_features) * 0.5 + 1)))
shap.summary_plot(shap_vals, X_te_woe[selected_features], show=False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_summary.png', bbox_inches='tight', dpi=120)
```

**Fig 5 — Expected vs Actual Decile Chart**
```python
tbl = exp_vs_act(y_te, te_proba)
plot_decile_chart(tbl, save_path=f'{OUTPUT_DIR}/figures/fig_decile.png')
tbl.to_csv(f'{OUTPUT_DIR}/data/table_decile.csv', index=False)
```

**Fig 6 — Score Distribution (defaults vs non-defaults)**
```python
fig, ax = plt.subplots(figsize=(9, 4))
ax.hist(te_proba[y_te == 0], bins=40, density=True, alpha=0.6,
        color='#5b9bd5', label='Non-Default (y=0)')
ax.hist(te_proba[y_te == 1], bins=40, density=True, alpha=0.6,
        color='#c00000', label='Default (y=1)')
ax.set_xlabel('Predicted Probability'); ax.set_ylabel('Density')
ax.set_title('Score Distribution — Test Set', fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_score_dist.png', bbox_inches='tight', dpi=120)
```

**Fig 7 — Feature Correlation Heatmap (WoE-encoded, train)**
```python
import seaborn as sns
corr = X_tr_woe[selected_features].corr()
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            mask=np.triu(np.ones_like(corr, dtype=bool), k=1),
            ax=ax, square=True, linewidths=0.5)
ax.set_title('Feature Correlation Matrix (WoE, Train)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_corr.png', bbox_inches='tight', dpi=120)
```

---

### Step 9 — Persist Outputs

```python
import pickle

bundle = {
    'model':       model.model_,
    'best_params': model.best_params_,
    'features':    selected_features,
    'ob_dict':     ob_dict,            # optbinning objects (WoE bins)
    'X_tr':        X_tr_woe[selected_features],
    'X_te':        X_te_woe[selected_features],
    'y_tr':        y_tr, 'y_te':    y_te,
    'tr_auc':      tr_auc, 'te_auc': te_auc,
    'tr_ks':       tr_ks,  'te_ks':  te_ks,
    'tr_proba':    tr_proba, 'te_proba': te_proba,
}
with open(f'{OUTPUT_DIR}/models/final_model.pkl', 'wb') as f:
    pickle.dump(bundle, f)
```

---

### Step 10 — Print Final Summary

```
=====================================
PROJECT  : {PROJECT_NAME}
AUTHOR   : {AUTHOR}
=====================================
Dataset  : {n} rows | {n_events} events ({pct:.1f}%) | ratio {ratio:.1f}:1
Features : {selected_features}
-------------------------------------
           AUC       KS
Train    : {tr_auc:.4f}   {tr_ks:.4f}
Test     : {te_auc:.4f}   {te_ks:.4f}
Gap      : {auc_gap:.4f}   {'✓ within 2pp' if auc_gap < 0.02 else '✗ EXCEEDS TARGET'}
-------------------------------------
Top decile captures {top_decile_pct:.1f}% of test-set events
=====================================
```

Top decile pct = events in highest-score decile / total test-set events × 100.

---

## Recommended Repository Structure

```
README.md
requirements.txt
index.html                    # Executive presentation (GitHub Pages root)
notebooks/
├── {PROJECT_NAME}.ipynb      # Main analysis notebook
└── run_pipeline.py           # Standalone end-to-end script
skills/
├── PD_MODEL_TEMPLATE.md      # This file
├── woe_iv.py                 # Fallback WoE/IV (no optbinning dependency)
├── model_utils.py            # Evaluation utilities
├── bayes_lgbm.py             # BayesLGBM with anti-overfitting penalty
└── nlp_txn_extractor.py      # Rule-based NLP extractor
paper/
└── {PROJECT_NAME}.pdf        # Research paper
models/
└── final_model.pkl           # Trained model bundle
figures/                      # All generated plots
data/                         # IV table, decile table (CSV)
scripts/
└── build_pdf.py              # Research paper build script
```

---

## Guard Rails

- Never refit WoE bins or any transformer on test data.
- IV ≥ 1.0 → flag as suspected leakage; do not auto-exclude without domain review.
- IV < 0.02 → exclude (no predictive signal).
- If imbalance ratio > 10:1, evaluate `scale_pos_weight` in LightGBM.
- If a time dimension exists, prefer chronological splitting over random splitting.
- Decision threshold of 0.50 is a starting point — calibrate against a cost matrix for production.
- Top decile pct must be calculated over test-set events only (not full dataset events).
- Do not use `optbinning` on Python 3.13+ without the patched fork at `pradark/optbinning-py313`.
- `n_jobs=1` in ShapRFECV avoids multiprocessing conflicts in some conda environments.
- When computing IV from optbinning's binning table, exclude the last 2 rows (totals/special).

---

## Skill Files

| File | Purpose |
|------|---------|
| `skills/woe_iv.py` | Fallback WoE/IV (pure pandas/numpy, no optbinning) |
| `skills/model_utils.py` | `performance_metrics`, `plot_auc2`, `exp_vs_act`, `plot_decile_chart` |
| `skills/bayes_lgbm.py` | `BayesLGBM` with anti-overfitting Bayesian optimisation |
| `skills/nlp_txn_extractor.py` | Configurable rule-based NLP extractor (130+ patterns) |

---

## Required Packages

```
# Core pipeline (optbinning_env — Python 3.11)
optbinning==0.21.0
lightgbm>=4.0
scikit-learn>=1.5
shap>=0.44
probatus>=3.1
bayesian-optimization>=1.5
pandas>=2.0
numpy>=1.26
scipy>=1.10
matplotlib>=3.8
seaborn>=0.12

# For PDF paper generation
reportlab>=4.0
pypdf>=4.0
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| optbinning segfault / protobuf crash on Python 3.13 ARM | ortools ABI incompatible | Use Python 3.11 conda env |
| `lightgbm` dylib error (`libomp.dylib not found`) on ARM Mac | Missing OpenMP | `cp /opt/anaconda3/lib/libomp.dylib $CONDA_ENV/lib/` |
| Categorical optbinning raises "Invalid monotonic trend" | `monotonic_trend` not valid for categorical | Omit the parameter for categorical dtype |
| `ShapRFECV` has no `get_results_cv()` in probatus ≥ 3.x | API changed | Use return value of `fit_compute()` directly |
| IV = 0 for binary features in optbinning | Near-degenerate bins collapse to 1 | Use `skills/woe_iv.py` for binary features as fallback |
| Top decile pct looks too low | Dividing by full-dataset events instead of test-set events | Denominator = `y_te.sum()` |
