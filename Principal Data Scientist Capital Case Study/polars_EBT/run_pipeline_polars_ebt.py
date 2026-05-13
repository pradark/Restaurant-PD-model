#!/usr/bin/env python3
"""
End-to-End PD Model Pipeline: Polars + Explainable Boosting Trees (EBT)
========================================================================
Follows 10-step methodology from PD_MODEL_TEMPLATE_POLARS_EBT.md
Uses Polars for data processing and EBT for interpretable predictions.
"""

import os
import sys
import pickle
import warnings
warnings.filterwarnings('ignore')

import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc
from optbinning import OptimalBinning
from interpret.glassbox import ExplainableBoostingClassifier
from probatus.feature_elimination import ShapRFECV
import lightgbm as lgb
import shap

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATHS = {
    'transactions': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_tx.csv",
    'accounts': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_account.csv",
    'labels': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_label.csv",
}

OUTPUT_DIR = "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT_REBUILD"
AUTHOR = "Data Science Team"
PROJECT_NAME = "Restaurant_PD_Model_Polars_EBT"
TARGET_COL = 'loan_default'

# Create output directories
for d in ['data', 'figures', 'models', 'reports']:
    Path(f'{OUTPUT_DIR}/{d}').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"PD MODEL PIPELINE: {PROJECT_NAME}")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}\n")

# ============================================================================
# STEP 1: DATA PROFILING (POLARS)
# ============================================================================

print("[STEP 1/10] Data Profiling with Polars...")

df_tx = pl.read_csv(DATA_PATHS['transactions'])
df_acc = pl.read_csv(DATA_PATHS['accounts'])
df_label = pl.read_csv(DATA_PATHS['labels'])

print(f"  ✓ Loaded transactions: {df_tx.shape}")
print(f"  ✓ Loaded accounts: {df_acc.shape}")
print(f"  ✓ Loaded labels: {df_label.shape}")

# Aggregate transactions to restaurant level
df_agg = (
    df_tx
    .group_by('Restaurant_ID')
    .agg([
        pl.col('processing_volume').sum().alias('total_volume'),
        pl.col('processing_volume').mean().alias('avg_volume'),
        pl.col('processing_volume').std().alias('std_volume'),
        pl.col('processing_volume').count().alias('num_tx'),
        pl.col('Tx_hours').mean().alias('avg_hours'),
        pl.col('Tx_hours').std().alias('std_hours'),
    ])
)

# Merge all data
df_all = (
    df_agg
    .join(df_acc, on='Restaurant_ID', how='left')
    .join(df_label, on='Restaurant_ID', how='left')
    .fill_null(0)
)

# Convert to pandas for sklearn compatibility
df = df_all.to_pandas()

print(f"  ✓ Final dataset shape: {df.shape}")
print(f"  ✓ Target distribution:\n{df[TARGET_COL].value_counts().to_dict()}")
print(f"  ✓ Missing values: {df.isnull().sum().sum()}")

# ============================================================================
# STEP 2: SKIP NLP (No text columns)
# ============================================================================

print("\n[STEP 2/10] NLP Feature Extraction... (SKIPPED - no text columns)")

# ============================================================================
# STEP 3: TRAIN/TEST SPLIT
# ============================================================================

print("\n[STEP 3/10] Train/Test Split (80/20 stratified)...")

X_all = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(
    X_all, y, test_size=0.2, stratify=y, random_state=42
)

print(f"  ✓ Training set: {X_tr_raw.shape} | Event rate: {y_tr.mean():.4f}")
print(f"  ✓ Test set: {X_te_raw.shape} | Event rate: {y_te.mean():.4f}")

# ============================================================================
# STEP 4: OPTBINNING WOE/IV
# ============================================================================

print("\n[STEP 4/10] OptBinning WoE/IV Feature Selection...")

def fit_optbinning(feature_name, x_train, y_train, dtype='numerical'):
    kwargs = dict(name=feature_name, dtype=dtype, solver='cp',
                  max_n_bins=10, min_bin_size=0.03)
    if dtype == 'numerical':
        kwargs['monotonic_trend'] = 'auto'
    
    ob = OptimalBinning(**kwargs)
    x = x_train.values.astype(float if dtype == 'numerical' else str)
    ob.fit(x, y_train.values)
    
    bt = ob.binning_table.build()
    iv = float(bt['IV'].iloc[:-2].sum()) if len(bt) > 2 else 0
    return ob, iv

numeric_features = X_tr_raw.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_tr_raw.select_dtypes(include=['object']).columns.tolist()

ob_dict = {}
iv_rows = []

for col in numeric_features + categorical_features:
    dtype = 'numerical' if col in numeric_features else 'categorical'
    x = X_tr_raw[col] if dtype == 'numerical' else X_tr_raw[col].fillna('Missing')
    
    try:
        ob, iv = fit_optbinning(col, x, y_tr, dtype=dtype)
        ob_dict[col] = ob
        iv_rows.append({'Feature': col, 'IV': round(iv, 4), 'Type': dtype})
    except Exception as e:
        print(f"    ⚠ Skipped {col}: {str(e)[:50]}")
        continue

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
print(f"  ✓ IV Table saved | Selected {len(selected_iv)} features")
print(f"    IV range: {iv_df['IV'].min():.4f} - {iv_df['IV'].max():.4f}")

# Apply WoE encoding
def apply_woe(X_df, ob_dict, features):
    out = pd.DataFrame(index=X_df.index)
    for col in features:
        if col not in ob_dict:
            continue
        ob = ob_dict[col]
        x = X_df[col].values.astype(float) if ob.dtype == 'numerical' \
            else X_df[col].fillna('Missing').values.astype(str)
        out[col] = ob.transform(x, metric='woe')
    return out.fillna(0.0)

X_tr_woe = apply_woe(X_tr_raw, ob_dict, selected_iv)
X_te_woe = apply_woe(X_te_raw, ob_dict, selected_iv)

print(f"  ✓ WoE encoding applied | Shape: {X_tr_woe.shape}")

# ============================================================================
# STEP 5: WOE PDP PLOTS
# ============================================================================

print("\n[STEP 5/10] WoE Partial Dependence Plots...")

numeric_selected = [f for f in selected_iv if f in numeric_features]
n_plots = len(numeric_selected)
nrows = (n_plots + 1) // 2
ncols = 2 if n_plots > 1 else 1

fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
if nrows == 1 and ncols == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, feat in enumerate(numeric_selected):
    if feat not in ob_dict:
        continue
    ax = axes[idx]
    ob = ob_dict[feat]
    bt = ob.binning_table.build().iloc[:-2]
    
    ax2 = ax.twinx()
    ax.bar(range(len(bt)), bt['Count (%)'], color='#5b9bd5', alpha=0.55, label='Count %')
    ax2.plot(range(len(bt)), bt['Event rate'], color='#c00000', marker='o', linewidth=2, label='Event Rate')
    ax.set_title(f'{feat}', fontweight='bold')
    ax.set_ylabel('Count (%)')
    ax2.set_ylabel('Event Rate')

# Remove extra subplots
for idx in range(len(numeric_selected), len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_woe_pdp.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_woe_pdp.png")

# ============================================================================
# STEP 6: SHAP RFE
# ============================================================================

print("\n[STEP 6/10] SHAP Recursive Feature Elimination...")

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

print(f"  ✓ Selected {len(selected_features)} features via SHAP RFE")
print(f"    Best CV AUC: {results_df['val_metric_mean'].max():.4f} at n={int(best_n)}")

# Plot elbow curve
fig, ax = plt.subplots(figsize=(10, 6))
ax.errorbar(results_df['num_features'], results_df['val_metric_mean'],
            yerr=results_df.get('val_metric_std', 0), fmt='o-', color='#1f4e79', capsize=5, linewidth=2)
ax.axvline(best_n, color='#c00000', linestyle='--', linewidth=2, label=f'Selected: {int(best_n)} features')
ax.set_xlabel('Number of Features', fontsize=11)
ax.set_ylabel('CV AUC', fontsize=11)
ax.set_title('SHAP RFE Elbow Curve', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_rfe.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_shap_rfe.png")

# ============================================================================
# STEP 7: TRAIN EBT MODEL
# ============================================================================

print("\n[STEP 7/10] Training Explainable Boosting Trees (EBT)...")

ebt_model = ExplainableBoostingClassifier(
    interactions=10,
    outer_bags=8,
    inner_bags=4,
    learning_rate=0.05,
    max_rounds=5000,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

ebt_model.fit(X_tr_woe[selected_features], y_tr)

# Predictions
tr_proba = ebt_model.predict_proba(X_tr_woe[selected_features])[:, 1]
te_proba = ebt_model.predict_proba(X_te_woe[selected_features])[:, 1]

tr_auc = roc_auc_score(y_tr, tr_proba)
te_auc = roc_auc_score(y_te, te_proba)
auc_gap = tr_auc - te_auc

print(f"  ✓ Model trained successfully")
print(f"    Train AUC: {tr_auc:.4f}")
print(f"    Test AUC:  {te_auc:.4f}")
print(f"    Gap:       {auc_gap:.4f} {'✓ stable' if auc_gap < 0.02 else '⚠ overfitting'}")

# ============================================================================
# STEP 8: EVALUATION FIGURES (7 figures)
# ============================================================================

print("\n[STEP 8/10] Generating Evaluation Figures...")

# Figure 1: ROC + KS Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# ROC
fpr_tr, tpr_tr, _ = roc_curve(y_tr, tr_proba)
fpr_te, tpr_te, _ = roc_curve(y_te, te_proba)

ax1.plot(fpr_tr, tpr_tr, label=f'Train (AUC={tr_auc:.4f})', linewidth=2)
ax1.plot(fpr_te, tpr_te, label=f'Test (AUC={te_auc:.4f})', linewidth=2)
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('ROC Curve', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# KS
ks_tr = np.max(tpr_tr - fpr_tr)
ks_te = np.max(tpr_te - fpr_te)

ax2.hist(tr_proba[y_tr == 0], bins=50, alpha=0.5, label='Train Non-Default', density=True)
ax2.hist(tr_proba[y_tr == 1], bins=50, alpha=0.5, label='Train Default', density=True)
ax2.axvline(tr_proba.quantile(0.9), color='red', linestyle='--', label=f'90th percentile')
ax2.set_xlabel('Predicted Probability')
ax2.set_ylabel('Density')
ax2.set_title(f'Score Distribution (Train) KS={ks_tr:.4f}', fontweight='bold')
ax2.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_roc_ks.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_roc_ks.png")

# Figure 2: Confusion Matrix
from sklearn.metrics import confusion_matrix
optimal_threshold = 0.5  # Can be tuned
y_pred = (te_proba >= optimal_threshold).astype(int)
cm = confusion_matrix(y_te, y_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['Non-Default', 'Default'],
            yticklabels=['Non-Default', 'Default'],
            annot_kws={'size': 14})
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')
ax.set_title('Confusion Matrix (Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_confusion.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_confusion.png")

# Figure 3: SHAP Bar Chart
explainer = shap.TreeExplainer(ebt_model.model_)
shap_vals = explainer.shap_values(X_te_woe[selected_features])
if isinstance(shap_vals, list):
    shap_vals = shap_vals[1]

mean_shap = np.abs(shap_vals).mean(axis=0)
shap_df = pd.DataFrame({'feature': selected_features, 'importance': mean_shap}).sort_values('importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, max(6, len(selected_features) * 0.4)))
ax.barh(shap_df['feature'], shap_df['importance'], color='#1f4e79')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Feature Importance (EBT, Test Set)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_bar.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_shap_bar.png")

# Figure 4: SHAP Summary
fig = plt.figure(figsize=(10, max(6, len(selected_features) * 0.4)))
shap.summary_plot(shap_vals, X_te_woe[selected_features], plot_type="dot", show=False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_shap_summary.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_shap_summary.png")

# Figure 5: Decile Chart
deciles = np.percentile(te_proba, np.linspace(0, 100, 11))
decile_bins = np.digitize(te_proba, deciles) - 1

decile_data = []
for d in range(10):
    mask = decile_bins == d
    if mask.sum() == 0:
        continue
    decile_data.append({
        'Decile': d + 1,
        'Count': mask.sum(),
        'Defaults': y_te[mask].sum(),
        'Avg_PD': te_proba[mask].mean(),
        'Actual_Rate': y_te[mask].mean(),
    })

decile_df = pd.DataFrame(decile_data)
decile_df.to_csv(f'{OUTPUT_DIR}/data/table_decile.csv', index=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Predicted vs Actual
ax1.plot(decile_df['Decile'], decile_df['Avg_PD'], 'o-', label='Predicted PD', linewidth=2, markersize=8)
ax1.plot(decile_df['Decile'], decile_df['Actual_Rate'], 's-', label='Actual Rate', linewidth=2, markersize=8)
ax1.set_xlabel('Decile')
ax1.set_ylabel('Default Probability')
ax1.set_title('Calibration by Decile', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Count by decile
ax2.bar(decile_df['Decile'], decile_df['Count'], color='#5b9bd5', alpha=0.7)
ax2.set_xlabel('Decile')
ax2.set_ylabel('Count')
ax2.set_title('Distribution by Decile', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_decile.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_decile.png")

# Figure 6: Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(te_proba[y_te == 0], bins=50, density=True, alpha=0.6,
        color='#5b9bd5', label='Non-Default (y=0)', edgecolor='black')
ax.hist(te_proba[y_te == 1], bins=50, density=True, alpha=0.6,
        color='#c00000', label='Default (y=1)', edgecolor='black')
ax.set_xlabel('Predicted Probability')
ax.set_ylabel('Density')
ax.set_title('Score Distribution — Test Set', fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_score_dist.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_score_dist.png")

# Figure 7: Correlation Heatmap
corr = X_tr_woe[selected_features].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True,
            mask=np.triu(np.ones_like(corr, dtype=bool), k=1),
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Correlation'})
ax.set_title('Feature Correlation Matrix (WoE, Train)', fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_corr.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_corr.png")

# ============================================================================
# STEP 9: PERSIST OUTPUTS
# ============================================================================

print("\n[STEP 9/10] Persisting Model and Outputs...")

bundle = {
    'model': ebt_model,
    'features': selected_features,
    'ob_dict': ob_dict,
    'X_tr': X_tr_woe[selected_features],
    'X_te': X_te_woe[selected_features],
    'y_tr': y_tr,
    'y_te': y_te,
    'tr_auc': tr_auc,
    'te_auc': te_auc,
    'tr_proba': tr_proba,
    'te_proba': te_proba,
}

with open(f'{OUTPUT_DIR}/models/final_model_ebt.pkl', 'wb') as f:
    pickle.dump(bundle, f)

print(f"  ✓ Model saved: {OUTPUT_DIR}/models/final_model_ebt.pkl")

# Save predictions
pred_df = pd.DataFrame({
    'test_index': X_te_raw.index,
    'predicted_probability': te_proba,
    'predicted_score_0_100': te_proba * 100,
    'actual_default': y_te.values,
})
pred_df.to_csv(f'{OUTPUT_DIR}/data/test_predictions_ebt.csv', index=False)
print(f"  ✓ Predictions saved: {OUTPUT_DIR}/data/test_predictions_ebt.csv")

# ============================================================================
# STEP 10: FINAL SUMMARY
# ============================================================================

print("\n[STEP 10/10] Final Summary")
print("=" * 80)
print(f"PROJECT:    {PROJECT_NAME}")
print(f"AUTHOR:     {AUTHOR}")
print("=" * 80)
print(f"Dataset:    {len(df)} rows | {y.sum()} events ({100*y.mean():.2f}%)")
print(f"Features:   {len(selected_features)} selected (from {len(iv_df)})")
print(f"Model:      Explainable Boosting Trees (EBT)")
print(f"Algorithm:  EBT (interactions=10, bags=8, learning_rate=0.05)")
print("-" * 80)
print(f"           AUC       KS")
print(f"Train:    {tr_auc:.4f}    {ks_tr:.4f}")
print(f"Test:     {te_auc:.4f}    {ks_te:.4f}")
print(f"Gap:      {auc_gap:.4f}    {'✓ stable' if auc_gap < 0.02 else '⚠ overfitting'}")
print("-" * 80)

top_decile_pct = 100 * y_te[te_proba >= te_proba.quantile(0.9)].sum() / y_te.sum()
print(f"Top decile captures {top_decile_pct:.1f}% of test-set defaults")
print(f"Mean predicted PD: {te_proba.mean():.4f} (actual: {y_te.mean():.4f})")
print(f"Median predicted PD: {np.median(te_proba):.4f}")
print("=" * 80)
print(f"\n✓ Pipeline completed successfully!")
print(f"✓ All outputs saved to: {OUTPUT_DIR}/")
print("=" * 80)

