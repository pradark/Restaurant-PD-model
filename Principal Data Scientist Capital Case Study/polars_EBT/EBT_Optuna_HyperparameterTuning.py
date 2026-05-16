"""
Optuna-based Hyperparameter Tuning for ExplainableBoostingClassifier
=====================================================================
Optimize EBT hyperparameters using Optuna's TPE sampler with cross-validation.
"""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
import polars as pl
from pathlib import Path
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SETUP: Load Data and Prepare Features
# ============================================================================
print('='*80)
print('OPTUNA HYPERPARAMETER TUNING FOR EXPLAINABLE BOOSTING TREES')
print('='*80)

base_path = Path('/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study')
OUTPUT_DIR = str(base_path / 'polars_EBT')

print('\nLoading RFECV-selected features...')
# Load pre-scaled RFECV data from numpy arrays
X_train_scaled = np.load(f'{OUTPUT_DIR}/X_train_scaled_rfecv.npy')
X_test_scaled = np.load(f'{OUTPUT_DIR}/X_test_scaled_rfecv.npy')
y_train = np.load(f'{OUTPUT_DIR}/y_train_rfecv.npy')
y_test = np.load(f'{OUTPUT_DIR}/y_test_rfecv.npy')

print(f'✓ Features loaded: {X_train_scaled.shape[0]:,} rows × {X_train_scaled.shape[1]} features')

# Convert y to pandas Series for compatibility with cross-validation
y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

print(f'✓ Train set: {X_train_scaled.shape[0]:,} rows × {X_train_scaled.shape[1]} features')
print(f'✓ Test set: {X_test_scaled.shape[0]:,} rows')
print(f'✓ Default rate: {100 * y_train.mean():.2f}%')

# ============================================================================
# DEFINE OPTUNA OBJECTIVE FUNCTION
# ============================================================================
print('\n' + '='*80)
print('HYPERPARAMETER SPACE DEFINITION')
print('='*80)

def objective(trial):
    """
    Optuna objective function for EBT hyperparameter tuning.

    Hyperparameters to tune:
    - interactions: Number of feature interaction terms (0-20)
    - outer_bags: Number of outer bags (2-16)
    - inner_bags: Number of inner bags (0-8)
    - learning_rate: Learning rate (0.001-0.2, log scale)
    - max_rounds: Maximum boosting rounds (100-10000, log scale)
    - max_leaves: Maximum number of leaves per tree (2-32)
    - min_samples_leaf: Minimum samples per leaf (1-50)
    """

    # Define hyperparameter space
    interactions = trial.suggest_int('interactions', 0, 20)
    outer_bags = trial.suggest_int('outer_bags', 2, 16)
    inner_bags = trial.suggest_int('inner_bags', 0, 8)
    learning_rate = trial.suggest_float('learning_rate', 0.001, 0.2, log=True)
    max_rounds = trial.suggest_int('max_rounds', 100, 10000, log=True)
    max_leaves = trial.suggest_int('max_leaves', 2, 32)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 50)

    # Perform 5-fold cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(cv.split(X_train_scaled, y_train)):
        X_tr = X_train_scaled[train_idx]
        X_val = X_train_scaled[val_idx]
        # Handle both numpy arrays and pandas Series
        if isinstance(y_train, np.ndarray):
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]
        else:
            y_tr = y_train.iloc[train_idx]
            y_val = y_train.iloc[val_idx]

        try:
            # Train EBT with suggested hyperparameters
            ebt = ExplainableBoostingClassifier(
                interactions=interactions,
                outer_bags=outer_bags,
                inner_bags=inner_bags,
                learning_rate=learning_rate,
                max_rounds=max_rounds,
                max_leaves=max_leaves,
                min_samples_leaf=min_samples_leaf,
                random_state=42,
                n_jobs=-1
            )

            ebt.fit(X_tr, y_tr)

            # Evaluate on validation fold
            y_val_pred = ebt.predict_proba(X_val)[:, 1]
            fold_auc = roc_auc_score(y_val, y_val_pred)
            cv_scores.append(fold_auc)

        except Exception as e:
            # Log the error for debugging
            print(f"Error in fold {fold}: {str(e)}")
            # Return 0 AUC for invalid parameter combinations
            return 0.0

    # Return mean CV AUC
    mean_cv_auc = np.mean(cv_scores)

    return mean_cv_auc


# ============================================================================
# CONFIGURE OPTUNA STUDY
# ============================================================================
print('\nOptuna Configuration:')
print('  - Sampler: TPE (Tree-structured Parzen Estimator)')
print('  - Pruner: Median Pruner')
print('  - Direction: Maximize (ROC-AUC)')
print('  - Optimization Metric: 5-Fold Cross-Validation AUC')

# Create study with TPE sampler
sampler = TPESampler(seed=42, n_startup_trials=10)
pruner = MedianPruner()

study = optuna.create_study(
    direction='maximize',
    sampler=sampler,
    pruner=pruner
)

# ============================================================================
# RUN OPTUNA OPTIMIZATION
# ============================================================================
print('\n' + '='*80)
print('RUNNING OPTUNA OPTIMIZATION')
print('='*80)

n_trials = 50
print(f'\nStarting optimization with {n_trials} trials...')
print('(This may take 20-30 minutes depending on your hardware)\n')

# Optimize
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

# ============================================================================
# DISPLAY RESULTS
# ============================================================================
print('\n' + '='*80)
print('OPTIMIZATION RESULTS')
print('='*80)

# Get best trial
best_trial = study.best_trial

print(f'\nBest CV AUC: {best_trial.value:.4f}')
print(f'\nBest Hyperparameters:')
print('-' * 60)
for key, value in sorted(best_trial.params.items()):
    if isinstance(value, float):
        print(f'  {key:20s}: {value:.6f}')
    else:
        print(f'  {key:20s}: {value}')

# ============================================================================
# TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================
print('\n' + '='*80)
print('TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS')
print('='*80)

best_params = best_trial.params

print('\nTraining Explainable Boosting Trees model...')

ebt_final = ExplainableBoostingClassifier(
    interactions=best_params['interactions'],
    outer_bags=best_params['outer_bags'],
    inner_bags=best_params['inner_bags'],
    learning_rate=best_params['learning_rate'],
    max_rounds=best_params['max_rounds'],
    max_leaves=best_params['max_leaves'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42,
    n_jobs=-1
)

ebt_final.fit(X_train_scaled, y_train)

# Generate predictions
y_pred_train = ebt_final.predict_proba(X_train_scaled)[:, 1]
y_pred_test = ebt_final.predict_proba(X_test_scaled)[:, 1]

# Calculate AUC
train_auc = roc_auc_score(y_train, y_pred_train)
test_auc = roc_auc_score(y_test, y_pred_test)
overfit_gap = train_auc - test_auc

print(f"\n✓ Model trained successfully with Optuna-optimized parameters")
print(f"  - Training AUC: {train_auc:.4f}")
print(f"  - Test AUC: {test_auc:.4f}")
print(f"  - Overfitting gap: {overfit_gap:.4f} ({'✓ Good' if overfit_gap < 0.05 else '⚠ Monitor'})")

# ============================================================================
# COMPARISON: BASELINE VS OPTIMIZED
# ============================================================================
print('\n' + '='*80)
print('BASELINE vs OPTIMIZED HYPERPARAMETERS')
print('='*80)

baseline_params = {
    'interactions': 10,
    'outer_bags': 8,
    'inner_bags': 4,
    'learning_rate': 0.05,
    'max_rounds': 5000,
}

print('\nBaseline Hyperparameters (Original):')
for key, value in baseline_params.items():
    print(f'  {key:20s}: {value}')

print('\nOptuna-Optimized Hyperparameters:')
for key, value in best_params.items():
    print(f'  {key:20s}: {value}')

print('\nOptimized vs Baseline:')
for key in baseline_params.keys():
    baseline_val = baseline_params[key]
    opt_val = best_params[key]
    if isinstance(opt_val, float):
        change = ((opt_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
        print(f'  {key:20s}: {baseline_val:>10} → {opt_val:>10.4f} ({change:+.1f}%)')
    else:
        change = ((opt_val - baseline_val) / baseline_val) * 100 if baseline_val != 0 else 0
        print(f'  {key:20s}: {baseline_val:>10} → {opt_val:>10} ({change:+.1f}%)')

# ============================================================================
# OPTUNA STUDY STATISTICS
# ============================================================================
print('\n' + '='*80)
print('OPTIMIZATION STATISTICS')
print('='*80)

print(f'\nTrials Summary:')
print(f'  - Total trials: {len(study.trials)}')
print(f'  - Completed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}')
print(f'  - Pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}')
print(f'  - Failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}')

print(f'\nPerformance Range:')
completed_values = [t.value for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if completed_values:
    print(f'  - Worst CV AUC: {min(completed_values):.4f}')
    print(f'  - Best CV AUC: {max(completed_values):.4f}')
    print(f'  - Mean CV AUC: {np.mean(completed_values):.4f}')
    print(f'  - Std Dev: {np.std(completed_values):.4f}')

# ============================================================================
# SAVE RESULTS
# ============================================================================
print('\n' + '='*80)
print('SAVING OPTIMIZATION RESULTS')
print('='*80)

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

output_file = f'{OUTPUT_DIR}/optuna_optimization_results.csv'

# Save optimization history
trials_df = study.trials_dataframe()
trials_df.to_csv(output_file, index=False)
print(f'\n✓ Optimization history saved: {output_file}')

# Save best parameters
best_params_file = f'{OUTPUT_DIR}/optuna_best_hyperparameters.txt'
with open(best_params_file, 'w') as f:
    f.write('OPTUNA OPTIMIZED HYPERPARAMETERS\n')
    f.write('='*60 + '\n\n')
    f.write(f'Best CV AUC: {best_trial.value:.4f}\n\n')
    f.write('Best Hyperparameters:\n')
    for key, value in sorted(best_params.items()):
        if isinstance(value, float):
            f.write(f'  {key}: {value:.6f}\n')
        else:
            f.write(f'  {key}: {value}\n')
    f.write(f'\nFinal Model Performance:\n')
    f.write(f'  Train AUC: {train_auc:.4f}\n')
    f.write(f'  Test AUC: {test_auc:.4f}\n')
    f.write(f'  Overfitting Gap: {overfit_gap:.4f}\n')

print(f'✓ Best parameters saved: {best_params_file}')

print('\n' + '='*80)
print('OPTUNA OPTIMIZATION COMPLETE')
print('='*80)
print(f'\n✓ Optimized model test AUC: {test_auc:.4f}')
print(f'✓ Results saved to feature_engineering_output/')
print(f'✓ Ready for production deployment')
