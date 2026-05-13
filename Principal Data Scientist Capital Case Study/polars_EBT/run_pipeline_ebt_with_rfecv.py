#!/usr/bin/env python3
"""
EBT PD Model Pipeline with RFE + Cross-Validation
===================================================
Complete 10-step implementation using EBT with RFECV for feature selection.
Avoids SHAP/probatus dependency conflicts by using sklearn's RFE directly.
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

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, auc, classification_report
from optbinning import OptimalBinning
from interpret.glassbox import ExplainableBoostingClassifier

# ============================================================================
# CONFIG
# ============================================================================

DATA_PATHS = {
    'transactions': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_tx.csv",
    'accounts': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_account.csv",
    'labels': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_label.csv",
}

OUTPUT_DIR = "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT_RFE_CV"
AUTHOR = "Data Science Team"
PROJECT_NAME = "Restaurant_PD_Model_EBT_RFECV"
TARGET_COL = 'loan_default'

# Create output directories
for d in ['data', 'figures', 'models', 'reports']:
    Path(f'{OUTPUT_DIR}/{d}').mkdir(parents=True, exist_ok=True)

print("=" * 80)
print(f"PD MODEL PIPELINE: {PROJECT_NAME}")
print("=" * 80)
print(f"Feature Selection: RFE with Cross-Validation (EBT)")
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
# STEP 2: SKIP NLP
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

print("\n[STEP 4/10] OptBinning WoE/IV Analysis...")

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
print(f"  ✓ IV Table saved | Pre-selected {len(selected_iv)} features (IV < 1.0)")
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

if n_plots > 0:
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

    for idx in range(len(numeric_selected), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/fig_woe_pdp.png', bbox_inches='tight', dpi=120)
    plt.close()
    print(f"  ✓ Saved: fig_woe_pdp.png")

# ============================================================================
# STEP 6: RFE WITH CROSS-VALIDATION (EBT)
# ============================================================================

print("\n[STEP 6/10] RFE with Cross-Validation (EBT estimator)...")

# Initialize EBT estimator for RFE
ebt_base = ExplainableBoostingClassifier(
    interactions=10,
    outer_bags=8,
    inner_bags=4,
    learning_rate=0.05,
    max_rounds=5000,
    random_state=42,
    n_jobs=-1
)

# Setup RFECV with 5-fold cross-validation
rfecv = RFECV(
    estimator=ebt_base,
    step=1,  # Remove 1 feature at a time
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

print("  Running RFECV (this may take 10-15 minutes)...")
rfecv.fit(X_tr_woe, y_tr)

selected_features = X_tr_woe.columns[rfecv.support_].tolist()
n_features = rfecv.n_features_

print(f"  ✓ RFECV completed")
print(f"  ✓ Selected {n_features} features (from {X_tr_woe.shape[1]} original)")
print(f"    Optimal features: {selected_features}")
print(f"    CV Scores (per round): min={rfecv.cv_results_['mean_test_score'].min():.4f}, "
      f"max={rfecv.cv_results_['mean_test_score'].max():.4f}")

# Plot RFECV results
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(rfecv.cv_results_['mean_test_score']) + 1), 
         rfecv.cv_results_['mean_test_score'], marker='o', color='#2E86AB', linewidth=2)
plt.fill_between(range(1, len(rfecv.cv_results_['mean_test_score']) + 1),
                 rfecv.cv_results_['mean_test_score'] - rfecv.cv_results_['std_test_score'],
                 rfecv.cv_results_['mean_test_score'] + rfecv.cv_results_['std_test_score'],
                 alpha=0.2, color='#2E86AB')
plt.xlabel('Number of Features Selected', fontsize=12, fontweight='bold')
plt.ylabel('Cross-Validated ROC-AUC Score', fontsize=12, fontweight='bold')
plt.title('RFECV: Feature Selection with EBT', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.axvline(n_features, color='red', linestyle='--', linewidth=2, label=f'Optimal: {n_features} features')
plt.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_rfecv_curve.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_rfecv_curve.png")

# Select final features for modeling
X_tr_final = X_tr_woe[selected_features]
X_te_final = X_te_woe[selected_features]

print(f"  ✓ Final dataset shape: {X_tr_final.shape}")

# ============================================================================
# STEP 7: TRAIN EBT MODEL
# ============================================================================

print("\n[STEP 7/10] Training Final EBT Model...")

ebt_model = ExplainableBoostingClassifier(
    interactions=10,
    outer_bags=8,
    inner_bags=4,
    learning_rate=0.05,
    max_rounds=5000,
    random_state=42,
    n_jobs=-1
)

ebt_model.fit(X_tr_final, y_tr)

y_tr_pred = ebt_model.predict(X_tr_final)
y_tr_proba = ebt_model.predict_proba(X_tr_final)[:, 1]
y_te_pred = ebt_model.predict(X_te_final)
y_te_proba = ebt_model.predict_proba(X_te_final)[:, 1]

auc_tr = roc_auc_score(y_tr, y_tr_proba)
auc_te = roc_auc_score(y_te, y_te_proba)

print(f"  ✓ Model trained successfully")
print(f"  ✓ Training AUC: {auc_tr:.4f}")
print(f"  ✓ Test AUC: {auc_te:.4f}")
print(f"  ✓ Overfitting gap: {(auc_tr - auc_te):.4f}")

# ============================================================================
# STEP 8: EVALUATION FIGURES
# ============================================================================

print("\n[STEP 8/10] Generating Evaluation Figures...")

# 1. ROC & KS Curves
fpr_tr, tpr_tr, _ = roc_curve(y_tr, y_tr_proba)
fpr_te, tpr_te, _ = roc_curve(y_te, y_te_proba)
ks_te = max(tpr_te - fpr_te)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.plot(fpr_tr, tpr_tr, label=f'Train (AUC={auc_tr:.4f})', linewidth=2.5, color='#2E86AB')
ax1.plot(fpr_te, tpr_te, label=f'Test (AUC={auc_te:.4f})', linewidth=2.5, color='#A23B72')
ax1.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
ax1.set_title('ROC Curves', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(tpr_te - fpr_te, linewidth=2.5, color='#F18F01', label=f'K-S = {ks_te:.4f}')
ax2.axhline(0, color='k', linestyle='-', alpha=0.2)
ax2.set_xlabel('Threshold Index', fontsize=11, fontweight='bold')
ax2.set_ylabel('K-S Statistic', fontsize=11, fontweight='bold')
ax2.set_title('Kolmogorov-Smirnov Curve', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_roc_ks.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_roc_ks.png")

# 2. Confusion Matrix
cm = confusion_matrix(y_te, y_te_pred)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False, 
            xticklabels=['Non-Default', 'Default'], yticklabels=['Non-Default', 'Default'])
ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
ax.set_title('Confusion Matrix (Test Set)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_confusion.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_confusion.png")

# 3. Feature Importance (EBT)
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': ebt_model.feature_importances_
}).sort_values('Importance', ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(feature_importance['Feature'].head(15), feature_importance['Importance'].head(15), color='#2E86AB')
ax.set_xlabel('Importance', fontsize=11, fontweight='bold')
ax.set_title('EBT Feature Importance (Top 15)', fontsize=12, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_ebt_importance.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_ebt_importance.png")

# 4. Score Distribution
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(y_te_proba[y_te == 0], bins=50, alpha=0.6, label='Non-Default', color='#2E86AB')
ax.hist(y_te_proba[y_te == 1], bins=50, alpha=0.6, label='Default', color='#A23B72')
ax.set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax.set_title('Predicted Score Distribution (Test Set)', fontsize=12, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_score_dist.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_score_dist.png")

# 5. Decile Analysis
deciles = pd.qcut(y_te_proba, q=10, labels=False) + 1
decile_analysis = pd.DataFrame({
    'Decile': deciles,
    'Default': y_te,
    'Pred_Prob': y_te_proba
}).groupby('Decile').agg({
    'Default': ['count', 'sum', 'mean'],
    'Pred_Prob': 'mean'
}).round(4)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

deciles_list = range(1, 11)
actual_rates = [y_te[deciles == d].mean() for d in deciles_list]
pred_rates = [y_te_proba[deciles == d].mean() for d in deciles_list]

ax1.plot(deciles_list, actual_rates, marker='o', linewidth=2, label='Actual Default Rate', color='#A23B72')
ax1.plot(deciles_list, pred_rates, marker='s', linewidth=2, label='Predicted Default Rate', color='#2E86AB')
ax1.set_xlabel('Decile', fontsize=11, fontweight='bold')
ax1.set_ylabel('Default Rate', fontsize=11, fontweight='bold')
ax1.set_title('Calibration by Decile', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

decile_counts = [y_te[deciles == d].shape[0] for d in deciles_list]
ax2.bar(deciles_list, decile_counts, color='#2E86AB', alpha=0.7)
ax2.set_xlabel('Decile', fontsize=11, fontweight='bold')
ax2.set_ylabel('Count', fontsize=11, fontweight='bold')
ax2.set_title('Observations per Decile', fontsize=12, fontweight='bold')
ax2.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_decile.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_decile.png")

# 6. Correlation Matrix
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(X_tr_final.corr(), cmap='coolwarm', center=0, ax=ax, square=True)
ax.set_title('Feature Correlation Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/fig_corr.png', bbox_inches='tight', dpi=120)
plt.close()
print(f"  ✓ Saved: fig_corr.png")

print(f"  ✓ All evaluation figures saved")

# ============================================================================
# STEP 9: MODEL PERSISTENCE
# ============================================================================

print("\n[STEP 9/10] Saving Model Bundle...")

model_bundle = {
    'model': ebt_model,
    'selected_features': selected_features,
    'rfecv': rfecv,
    'ob_dict': ob_dict,
    'feature_importance': feature_importance,
    'metadata': {
        'project': PROJECT_NAME,
        'author': AUTHOR,
        'n_features': n_features,
        'auc_test': auc_te,
        'auc_train': auc_tr,
    }
}

with open(f'{OUTPUT_DIR}/models/final_model_ebt_rfecv.pkl', 'wb') as f:
    pickle.dump(model_bundle, f)

print(f"  ✓ Model saved: final_model_ebt_rfecv.pkl")

# ============================================================================
# STEP 10: SUMMARY STATISTICS
# ============================================================================

print("\n[STEP 10/10] Final Summary...")

print("\n" + "=" * 80)
print("MODEL PERFORMANCE SUMMARY")
print("=" * 80)

print(f"\nDiscrimination Metrics:")
print(f"  • AUC (Test):        {auc_te:.4f} ({'GOOD' if auc_te > 0.75 else 'FAIR' if auc_te > 0.70 else 'POOR'})")
print(f"  • AUC (Train):       {auc_tr:.4f}")
print(f"  • K-S Statistic:     {ks_te:.4f}")
print(f"  • Overfitting Gap:   {(auc_tr - auc_te):.4f}")

tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
precision = tp / (tp + fp)
accuracy = (tp + tn) / (tp + tn + fp + fn)

print(f"\nClassification Performance:")
print(f"  • Sensitivity:       {sensitivity:.4f} ({sensitivity*100:.2f}%)")
print(f"  • Specificity:       {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  • Precision:         {precision:.4f}")
print(f"  • Accuracy:          {accuracy:.4f}")

print(f"\nFeature Selection (RFECV + EBT):")
print(f"  • Initial Features:  {X_tr_woe.shape[1]}")
print(f"  • Selected Features: {n_features}")
print(f"  • Reduction:         {(1 - n_features/X_tr_woe.shape[1])*100:.1f}%")

print(f"\nData Distribution:")
print(f"  • Training N:        {X_tr_final.shape[0]:,}")
print(f"  • Test N:            {X_te_final.shape[0]:,}")
print(f"  • Training Events:   {y_tr.sum():,} ({y_tr.mean()*100:.2f}%)")
print(f"  • Test Events:       {y_te.sum():,} ({y_te.mean()*100:.2f}%)")

print(f"\nProduction Readiness:")
auc_pass = "✓ PASS" if auc_te >= 0.75 else "✗ FAIL"
sens_pass = "✓ PASS" if sensitivity >= 0.40 else "✗ FAIL"
spec_pass = "✓ PASS" if specificity >= 0.70 else "✗ FAIL"

print(f"  • AUC >= 0.75:       {auc_pass}")
print(f"  • Sensitivity >= 40%: {sens_pass}")
print(f"  • Specificity >= 70%: {spec_pass}")

print(f"\n" + "=" * 80)
print(f"Output Directory: {OUTPUT_DIR}")
print(f"=" * 80 + "\n")
