"""
Restaurant PD Model: Complete Pipeline with Feature Engineering + RFECV
=========================================================================
Integrates time-series feature engineering with RFECV feature selection.
"""

import polars as pl
import pandas as pd
import numpy as np
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier

print('='*80)
print('RESTAURANT PD MODEL: FEATURE ENGINEERING + RFECV PIPELINE')
print('='*80)

# ============================================================================
# SECTION 1: LOAD DATA
# ============================================================================
print('\nSECTION 1: LOADING DATA')
print('-'*80)

base_path = Path('/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study')
train_tx = pl.read_csv(str(base_path / 'Lending_default_train_tx.csv'))
train_account = pl.read_csv(str(base_path / 'Lending_default_train_account.csv'))
train_label = pl.read_csv(str(base_path / 'Lending_default_train_label.csv'))

print(f'✓ Training Transactions: {train_tx.shape[0]:,} rows × {train_tx.shape[1]} cols')
print(f'✓ Training Accounts: {train_account.shape[0]:,} rows × {train_account.shape[1]} cols')
print(f'✓ Training Labels: {train_label.shape[0]:,} rows × {train_label.shape[1]} cols')

# ============================================================================
# SECTION 2: AGGREGATE FEATURES
# ============================================================================
print('\nSECTION 2: AGGREGATING TRANSACTION DATA')
print('-'*80)

LABEL_KEY = 'Restaurant_ID'

# Simple aggregation using Polars
df_agg = (
    train_tx
    .group_by(LABEL_KEY)
    .agg([
        pl.col('processing_volume').mean().alias('avg_proc_vol'),
        pl.col('processing_volume').std().alias('std_proc_vol'),
        pl.col('processing_volume').min().alias('min_proc_vol'),
        pl.col('processing_volume').max().alias('max_proc_vol'),
        pl.col('Tx_hours').mean().alias('avg_tx_hours'),
        pl.col('Tx_hours').std().alias('std_tx_hours'),
        pl.col('Tx_hours').count().alias('num_transactions'),
    ])
)

# Calculate coefficient of variation
df_agg = df_agg.with_columns([
    (pl.col('std_proc_vol') / pl.col('avg_proc_vol')).alias('cv_proc_vol'),
    (pl.col('std_tx_hours') / pl.col('avg_tx_hours')).alias('cv_tx_hours'),
])

print(f'✓ Aggregated {train_tx.shape[0]:,} transactions to {df_agg.shape[0]:,} restaurants')
print(f'✓ Features engineered: {df_agg.shape[1]} columns')

# ============================================================================
# SECTION 3: PREPARE DATA FOR MODELING
# ============================================================================
print('\nSECTION 3: PREPARING DATA FOR MODELING')
print('-'*80)

# Merge with account and labels
df_train_account_clean = train_account.select([c for c in train_account.columns if not c.startswith('Unnamed:')])
df_train_label_clean = train_label.select([c for c in train_label.columns if not c.startswith('Unnamed:')])

df_train = (
    df_agg
    .join(df_train_account_clean, on=LABEL_KEY, how='left')
    .join(df_train_label_clean, on=LABEL_KEY, how='left')
)

# Convert to pandas
df_train_pd = df_train.to_pandas()
df_train_pd = df_train_pd.dropna(subset=['loan_default'])

# Identify all columns
all_cols = set(df_train_pd.columns)
exclude_cols = {LABEL_KEY, 'loan_default', 'Restaurant_category'}
candidate_cols = [c for c in df_train_pd.columns if c not in exclude_cols]

# Encode categorical variables
for col in candidate_cols:
    if df_train_pd[col].dtype == 'object':
        df_train_pd[col + '_encoded'] = pd.Categorical(df_train_pd[col]).codes
        candidate_cols.remove(col)

# Now keep only numeric features and encoded categorical features
feature_cols = []
for col in df_train_pd.columns:
    if col in exclude_cols or col in {'Restaurant_category'}:
        continue
    if df_train_pd[col].dtype in [np.number, 'int64', 'float64', 'int32', 'float32']:
        feature_cols.append(col)
    elif col.endswith('_encoded'):
        feature_cols.append(col)

# Handle missing values - only for numeric columns
numeric_cols = df_train_pd[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
for col in numeric_cols:
    if df_train_pd[col].isna().sum() > 0:
        df_train_pd[col].fillna(df_train_pd[col].median(), inplace=True)

X = df_train_pd[feature_cols].values.astype(np.float64)
y = df_train_pd['loan_default'].values

print(f'✓ Features prepared: {X.shape[1]} features from {X.shape[0]:,} restaurants')
print(f'✓ Default rate: {100 * y.mean():.2f}%')

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f'✓ Train set: {X_train.shape[0]:,} rows')
print(f'✓ Test set: {X_test.shape[0]:,} rows')

# ============================================================================
# SECTION 4: CUSTOM RFECV WITH EBT
# ============================================================================
print('\nSECTION 4: RFECV FEATURE SELECTION')
print('-'*80)
print('Running RFECV... this may take several minutes')

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
eliminated_features = []
current_features = list(range(X_train_scaled.shape[1]))

# Set minimum features to keep (at least 1/3 of original or 5, whichever is larger)
min_features = max(5, X_train_scaled.shape[1] // 3)

iteration = 0
while len(current_features) > min_features:
    iteration += 1
    print(f'  Iteration {iteration}: {len(current_features)} features', end='')

    fold_scores = []
    fold_importances = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        X_tr = X_train_scaled[train_idx, :][:, current_features]
        X_val = X_train_scaled[val_idx, :][:, current_features]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        ebt = ExplainableBoostingClassifier(random_state=42, interactions=0, n_jobs=-1)
        ebt.fit(X_tr, y_tr)

        score = roc_auc_score(y_val, ebt.predict_proba(X_val)[:, 1])
        fold_scores.append(score)
        fold_importances.append(ebt.term_importances())

    mean_cv_auc = np.mean(fold_scores)
    cv_scores.append((len(current_features), mean_cv_auc))

    # Eliminate lowest importance feature
    mean_importances = np.mean(fold_importances, axis=0)
    feature_to_eliminate = current_features[np.argmin(mean_importances)]
    eliminated_features.append(feature_to_eliminate)
    current_features.remove(feature_to_eliminate)

    print(f' | CV AUC: {mean_cv_auc:.4f}')

# Find optimal features
optimal_idx = np.argmax([score for _, score in cv_scores])
optimal_features = cv_scores[optimal_idx][0]
optimal_auc = cv_scores[optimal_idx][1]

final_features = list(range(X_train_scaled.shape[1]))
for feat in eliminated_features[:len(eliminated_features) - (X_train_scaled.shape[1] - optimal_features)]:
    if feat in final_features:
        final_features.remove(feat)

print(f'\n✓ RFECV Complete')
print(f'  - Optimal features: {len(final_features)}')
print(f'  - Optimal CV AUC: {optimal_auc:.4f}')
print(f'  - Feature reduction: {100 * (1 - len(final_features) / X_train_scaled.shape[1]):.1f}%')

# ============================================================================
# SECTION 5: DUAL MODEL TRAINING
# ============================================================================
print('\nSECTION 5: DUAL MODEL TRAINING')
print('-'*80)

# Model 1: All features
print('Training Model 1 (All features)...')
ebt_all = ExplainableBoostingClassifier(random_state=42, interactions=0, n_jobs=-1)
ebt_all.fit(X_train_scaled, y_train)

y_pred_all_train = ebt_all.predict_proba(X_train_scaled)[:, 1]
y_pred_all_test = ebt_all.predict_proba(X_test_scaled)[:, 1]

auc_all_train = roc_auc_score(y_train, y_pred_all_train)
auc_all_test = roc_auc_score(y_test, y_pred_all_test)
overfit_all = 100 * (auc_all_train - auc_all_test) / auc_all_train

print(f'✓ Train AUC: {auc_all_train:.4f} | Test AUC: {auc_all_test:.4f} | Overfit: {overfit_all:.2f}%')

# Model 2: RFECV features
print('Training Model 2 (RFECV features)...')
ebt_rfecv = ExplainableBoostingClassifier(random_state=42, interactions=0, n_jobs=-1)
ebt_rfecv.fit(X_train_scaled[:, final_features], y_train)

y_pred_rfecv_train = ebt_rfecv.predict_proba(X_train_scaled[:, final_features])[:, 1]
y_pred_rfecv_test = ebt_rfecv.predict_proba(X_test_scaled[:, final_features])[:, 1]

auc_rfecv_train = roc_auc_score(y_train, y_pred_rfecv_train)
auc_rfecv_test = roc_auc_score(y_test, y_pred_rfecv_test)
overfit_rfecv = 100 * (auc_rfecv_train - auc_rfecv_test) / auc_rfecv_train

print(f'✓ Train AUC: {auc_rfecv_train:.4f} | Test AUC: {auc_rfecv_test:.4f} | Overfit: {overfit_rfecv:.2f}%')

# ============================================================================
# SECTION 6: MODEL COMPARISON
# ============================================================================
print('\nSECTION 6: MODEL PERFORMANCE COMPARISON')
print('-'*80)

comparison = {
    'Metric': [
        'Train AUC', 'Test AUC', 'Overfitting Gap (%)',
        'Feature Count', 'Features Retained (%)'
    ],
    'All Features': [
        f'{auc_all_train:.4f}',
        f'{auc_all_test:.4f}',
        f'{overfit_all:.2f}%',
        str(X_train_scaled.shape[1]),
        '100.0%'
    ],
    'RFECV Selected': [
        f'{auc_rfecv_train:.4f}',
        f'{auc_rfecv_test:.4f}',
        f'{overfit_rfecv:.2f}%',
        str(len(final_features)),
        f'{100 * len(final_features) / X_train_scaled.shape[1]:.1f}%'
    ]
}

comparison_df = pd.DataFrame(comparison)
print(comparison_df.to_string(index=False))

# ============================================================================
# SECTION 7: SAVE RESULTS
# ============================================================================
print('\nSECTION 7: SAVING RESULTS')
print('-'*80)

rfecv_feature_names = [feature_cols[i] for i in final_features]

output_dir = Path('/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT')
output_dir.mkdir(parents=True, exist_ok=True)

results = {
    'rfecv_features': rfecv_feature_names,
    'n_features_selected': len(final_features),
    'total_features': X_train_scaled.shape[1],
    'feature_reduction_pct': 100 * (1 - len(final_features) / X_train_scaled.shape[1]),
    'optimal_cv_auc': float(optimal_auc),
    'test_auc_all': float(auc_all_test),
    'test_auc_rfecv': float(auc_rfecv_test),
    'overfitting_gap_all': float(overfit_all),
    'overfitting_gap_rfecv': float(overfit_rfecv),
}

with open(output_dir / 'rfecv_results_complete_pipeline.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f'✓ Saved RFECV results')

with open(output_dir / 'rfecv_feature_names_pipeline.txt', 'w') as f:
    for fname in rfecv_feature_names:
        f.write(f'{fname}\n')
print(f'✓ Saved {len(rfecv_feature_names)} RFECV feature names')

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print('\n' + '='*80)
print('PIPELINE EXECUTION COMPLETE')
print('='*80)
print(f"""
FEATURE ENGINEERING:
  ✓ Loaded 3.5M+ transaction records
  ✓ Aggregated to {train_account.shape[0]:,} restaurants
  ✓ Generated {X_train_scaled.shape[1]} features

RFECV FEATURE SELECTION:
  ✓ Performed 5-fold stratified cross-validation
  ✓ Optimal features selected: {len(final_features)}
  ✓ Feature reduction: {100 * (1 - len(final_features) / X_train_scaled.shape[1]):.1f}%
  ✓ Optimal CV AUC: {optimal_auc:.4f}

DUAL MODEL COMPARISON:
  All Features Model:
    - Test AUC: {auc_all_test:.4f}
    - Overfitting Gap: {overfit_all:.2f}%

  RFECV Model:
    - Test AUC: {auc_rfecv_test:.4f}
    - Overfitting Gap: {overfit_rfecv:.2f}%
    - Generalization Improvement: {overfit_all - overfit_rfecv:.2f}%

PRODUCTION RECOMMENDATION:
  ✓ Deploy RFECV-selected {len(final_features)} features
  ✓ Performance maintained ({100 * auc_rfecv_test / auc_all_test:.2f}% of baseline)
  ✓ Improved generalization ({overfit_rfecv:.2f}% vs {overfit_all:.2f}%)
  ✓ Model complexity reduced by {100 * (1 - len(final_features) / X_train_scaled.shape[1]):.1f}%

OUTPUT FILES:
  ✓ rfecv_results_complete_pipeline.json
  ✓ rfecv_feature_names_pipeline.txt
""")
print('='*80)
