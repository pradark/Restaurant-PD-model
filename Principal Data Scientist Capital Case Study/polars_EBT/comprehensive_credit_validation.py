#!/usr/bin/env python3
"""
Comprehensive Credit Risk Model Validation
Rebuilds the EBT model with proper train-test split and calculates production-grade statistics
"""

import pandas as pd
import numpy as np
import polars as pl
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_recall_curve, f1_score, matthews_corrcoef
)
from interpret.glassbox import ExplainableBoostingClassifier
import sys
import os

warnings.filterwarnings('ignore')

# Set base path
BASE_PATH = "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study"

print("=" * 80)
print("COMPREHENSIVE CREDIT RISK VALIDATION - EBT MODEL")
print("=" * 80)

try:
    # ========================================================================
    # STEP 1: LOAD AND AGGREGATE DATA
    # ========================================================================
    print("\n[STEP 1/6] Loading training data...")
    
    df_train_tx = pl.read_csv(os.path.join(BASE_PATH, "Lending_default_train_tx.csv"))
    df_train_acc = pl.read_csv(os.path.join(BASE_PATH, "Lending_default_train_account.csv"))
    df_train_label = pl.read_csv(os.path.join(BASE_PATH, "Lending_default_train_label.csv"))
    
    print(f"  ✓ Loaded transaction data: {df_train_tx.shape}")
    print(f"  ✓ Loaded account data: {df_train_acc.shape}")
    print(f"  ✓ Loaded labels: {df_train_label.shape}")
    
    # ========================================================================
    # STEP 2: FEATURE ENGINEERING (Rolling Windows)
    # ========================================================================
    print("\n[STEP 2/6] Engineering rolling time-series features...")
    
    # Convert to pandas for aggregation
    tx_pandas = df_train_tx.to_pandas().sort_values('Tx_date')
    
    # Create rolling windows
    rolling_features = []
    for rest_id in tx_pandas['Restaurant_ID'].unique():
        rest_data = tx_pandas[tx_pandas['Restaurant_ID'] == rest_id].sort_values('Tx_date')
        
        if len(rest_data) == 0:
            continue
        
        last_date = rest_data['Tx_date'].max()
        
        # 7-day, 30-day, 120-day, 180-day lookbacks
        lookback_days = {
            '7d': 7,
            '30d': 30,
            '120d': 120,
            '180d': 180
        }
        
        for period, days in lookback_days.items():
            cutoff = pd.Timestamp(last_date) - pd.Timedelta(days=days)
            period_data = rest_data[pd.to_datetime(rest_data['Tx_date']) >= cutoff]
            
            rolling_features.append({
                'Restaurant_ID': rest_id,
                f'volume_{period}': period_data['processing_volume'].sum(),
                f'count_{period}': len(period_data),
                f'avg_vol_{period}': period_data['processing_volume'].mean(),
                f'std_vol_{period}': period_data['processing_volume'].std(),
            })
    
    rolling_df = pd.DataFrame(rolling_features)
    rolling_df = rolling_df.groupby('Restaurant_ID').first().reset_index()
    rolling_df = rolling_df.fillna(0)
    
    # Aggregate transaction data
    df_train_agg = (
        df_train_tx
        .group_by('Restaurant_ID')
        .agg([
            pl.col('processing_volume').sum().alias('total_volume'),
            pl.col('processing_volume').mean().alias('avg_volume'),
            pl.col('processing_volume').std().alias('std_volume'),
            pl.col('processing_volume').count().alias('num_transactions'),
            pl.col('Tx_hours').mean().alias('avg_tx_hours'),
        ])
        .to_pandas()
    )
    
    # Merge all features
    df_train_acc_pandas = df_train_acc.to_pandas()
    df_train_label_pandas = df_train_label.to_pandas()
    
    df_final = (
        df_train_agg
        .merge(df_train_acc_pandas, on='Restaurant_ID')
        .merge(df_train_label_pandas, on='Restaurant_ID')
        .merge(rolling_df, on='Restaurant_ID', how='left')
    )
    
    df_final = df_final.fillna(0)
    
    print(f"  ✓ Final dataset shape: {df_final.shape}")
    print(f"  ✓ Default rate: {df_final['loan_default'].mean():.4f}")
    
    # ========================================================================
    # STEP 3: PREPARE DATA FOR MODELING
    # ========================================================================
    print("\n[STEP 3/6] Preparing data for modeling...")
    
    # Select features
    categorical_cols = ['Ownership_type', 'Restaurant_catagory', 'Market_segment']
    numeric_cols = [col for col in df_final.columns 
                   if col not in ['Restaurant_ID', 'loan_default'] + categorical_cols]
    
    # Encode categorical variables
    df_encoded = pd.get_dummies(df_final[categorical_cols + numeric_cols], 
                                columns=categorical_cols, drop_first=False)
    
    X = df_encoded
    y = df_final['loan_default'].values
    
    # Stratified train-test split
    X_train, X_test, y_train, y_test, rest_id_train, rest_id_test = train_test_split(
        X, y, df_final['Restaurant_ID'].values,
        test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"  ✓ Training set size: {X_train.shape}")
    print(f"  ✓ Test set size: {X_test.shape}")
    print(f"  ✓ Feature count: {X_train.shape[1]}")
    print(f"  ✓ Train event rate: {y_train.mean():.4f}")
    print(f"  ✓ Test event rate: {y_test.mean():.4f}")
    
    # ========================================================================
    # STEP 4: TRAIN EXPLAINABLE BOOSTING TREES
    # ========================================================================
    print("\n[STEP 4/6] Training Explainable Boosting Trees (EBT)...")
    
    ebt = ExplainableBoostingClassifier(
        interactions=10,
        outer_bags=8, 
        inner_bags=4,
        learning_rate=0.05,
        max_rounds=5000,
        random_state=42
    )
    ebt.fit(X_train, y_train)
    
    y_pred_train = ebt.predict_proba(X_train)[:, 1]
    y_pred_test = ebt.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_pred_train)
    test_auc = roc_auc_score(y_test, y_pred_test)
    
    print(f"  ✓ Training AUC: {train_auc:.4f}")
    print(f"  ✓ Test AUC: {test_auc:.4f}")
    
    # ========================================================================
    # STEP 5: COMPREHENSIVE VALIDATION METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION METRICS")
    print("=" * 80)
    
    # 5.1 Discrimination Metrics
    print("\n1. DISCRIMINATION METRICS")
    print("   (Ability to separate defaults from non-defaults)")
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_test)
    ks_stat = np.max(tpr - fpr)
    gini = 2 * test_auc - 1
    
    print(f"   AUC (Area Under Curve):      {test_auc:.4f}")
    print(f"   Gini Coefficient:            {gini:.4f}")
    print(f"   K-S Statistic:               {ks_stat:.4f}")
    
    print(f"\n   Credit Risk Assessment:")
    if test_auc < 0.65:
        rating = "POOR - Model unsuitable for lending"
    elif test_auc < 0.70:
        rating = "WEAK - High risk, not recommended"
    elif test_auc < 0.75:
        rating = "MARGINAL - Monitor closely, additional validation needed"
    elif test_auc < 0.80:
        rating = "ACCEPTABLE - Fair discriminatory power"
    elif test_auc < 0.85:
        rating = "GOOD - Suitable for credit decisions"
    else:
        rating = "EXCELLENT - Strong predictive power"
    
    print(f"   Rating: {rating}")
    
    # 5.2 Calibration Analysis
    print("\n2. CALIBRATION ANALYSIS")
    print("   (Do predicted probabilities match actual default rates?)\n")
    
    n_deciles = 10
    deciles = np.percentile(y_pred_test, np.linspace(0, 100, n_deciles + 1))
    decile_groups = np.digitize(y_pred_test, deciles) - 1
    
    print(f"{'Decile':<8} {'N':<8} {'Defaults':<10} {'Predicted_PD':<14} {'Actual_Rate':<12} {'MAE':<8}")
    print("-" * 70)
    
    calibration_errors = []
    for decile in range(n_deciles):
        mask = decile_groups == decile
        if mask.sum() == 0:
            continue
        
        n_obs = mask.sum()
        n_def = y_test[mask].sum()
        avg_pred = y_pred_test[mask].mean()
        actual_rate = y_test[mask].mean()
        mae = abs(actual_rate - avg_pred)
        calibration_errors.append(mae)
        
        print(f"{decile+1:<8} {n_obs:<8} {int(n_def):<10} {avg_pred:<14.4f} {actual_rate:<12.4f} {mae:<8.4f}")
    
    mean_cal_error = np.mean(calibration_errors)
    print("-" * 70)
    print(f"\nMean Absolute Calibration Error: {mean_cal_error:.4f}")
    
    if mean_cal_error < 0.05:
        cal_rating = "EXCELLENT - Predictions are well-calibrated"
    elif mean_cal_error < 0.10:
        cal_rating = "GOOD - Acceptable calibration"
    elif mean_cal_error < 0.15:
        cal_rating = "FAIR - Moderate calibration error"
    else:
        cal_rating = "POOR - Systematic calibration bias"
    
    print(f"Calibration Rating: {cal_rating}")
    
    # 5.3 Classification Metrics
    print("\n3. CLASSIFICATION PERFORMANCE")
    
    # Optimal threshold
    precision_vals, recall_vals, threshold_vals = precision_recall_curve(y_test, y_pred_test)
    f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = threshold_vals[optimal_idx] if optimal_idx < len(threshold_vals) else 0.5
    
    y_pred_binary = (y_pred_test >= optimal_threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    mcc = matthews_corrcoef(y_test, y_pred_binary)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    print(f"   Optimal Threshold: {optimal_threshold:.4f}\n")
    print(f"   Confusion Matrix:")
    print(f"   {'':20} Predicted Negative  Predicted Positive")
    print(f"   {'Actual Negative':<20} {tn:<19} {fp:<19}")
    print(f"   {'Actual Positive':<20} {fn:<19} {tp:<19}\n")
    
    print(f"   Sensitivity (Default Detection):     {sensitivity:.4f} (Catch {100*sensitivity:.1f}% of defaults)")
    print(f"   Specificity (Non-Default Precision): {specificity:.4f} (Avoid {100*specificity:.1f}% false alarms)")
    print(f"   Precision (Positive Predictive Val): {precision:.4f} ({100*precision:.1f}% of flagged are defaults)")
    print(f"   Recall (True Positive Rate):         {recall:.4f}")
    print(f"   Accuracy:                            {accuracy:.4f}")
    print(f"   F1 Score:                            {f1:.4f}")
    print(f"   Matthews Correlation Coeff:          {mcc:.4f}")
    
    # 5.4 Segment Analysis
    print("\n4. SEGMENT-LEVEL PERFORMANCE")
    
    # Create test dataframe with predictions
    test_indices = X_test.index
    test_df = df_final.iloc[test_indices].copy()
    test_df['pred_prob'] = y_pred_test
    test_df['actual'] = y_test
    
    segments_to_analyze = ['Ownership_type', 'Restaurant_catagory', 'Market_segment']
    
    for segment in segments_to_analyze:
        if segment not in test_df.columns:
            continue
        
        print(f"\n   By {segment}:")
        seg_analysis = test_df.groupby(segment).agg({
            'actual': ['sum', 'count', 'mean'],
            'pred_prob': 'mean'
        }).round(4)
        
        seg_analysis.columns = ['Defaults', 'Total', 'Default_Rate', 'Avg_Pred_PD']
        
        # Calculate AUC by segment
        for seg_value in seg_analysis.index:
            seg_mask = test_df[segment] == seg_value
            if test_df.loc[seg_mask, 'actual'].nunique() > 1 and seg_mask.sum() > 10:
                seg_auc = roc_auc_score(test_df.loc[seg_mask, 'actual'], 
                                       test_df.loc[seg_mask, 'pred_prob'])
                seg_analysis.loc[seg_value, 'AUC'] = seg_auc
        
        print(seg_analysis.to_string())
    
    # 5.5 Risk Concentration
    print("\n5. RISK CONCENTRATION METRICS")
    
    high_risk = (y_pred_test > 0.30).sum()
    extreme_risk = (y_pred_test > 0.50).sum()
    
    print(f"   High-Risk Loans (PD > 30%):     {high_risk} ({100*high_risk/len(y_pred_test):.2f}%)")
    print(f"   Extreme-Risk Loans (PD > 50%):  {extreme_risk} ({100*extreme_risk/len(y_pred_test):.2f}%)")
    print(f"   Max Individual PD:               {y_pred_test.max():.4f}")
    print(f"   Mean PD:                         {y_pred_test.mean():.4f}")
    print(f"   Median PD:                       {np.median(y_pred_test):.4f}")
    
    # 5.6 Final Production Readiness
    print("\n" + "=" * 80)
    print("6. PRODUCTION READINESS CHECKLIST")
    print("=" * 80)
    
    checklist = []
    
    # Discrimination
    if test_auc >= 0.75:
        checklist.append(("Discrimination (AUC >= 0.75)", "✓ PASS", test_auc))
    else:
        checklist.append(("Discrimination (AUC >= 0.75)", "✗ FAIL", test_auc))
    
    # Calibration
    if mean_cal_error < 0.10:
        checklist.append(("Calibration (MAE < 0.10)", "✓ PASS", mean_cal_error))
    else:
        checklist.append(("Calibration (MAE < 0.10)", "✗ FAIL", mean_cal_error))
    
    # Default Detection
    if sensitivity >= 0.40:
        checklist.append(("Default Detection (Sensitivity >= 40%)", "✓ PASS", sensitivity))
    else:
        checklist.append(("Default Detection (Sensitivity >= 40%)", "✗ FAIL", sensitivity))
    
    # Risk Avoidance
    if specificity >= 0.70:
        checklist.append(("Risk Avoidance (Specificity >= 70%)", "✓ PASS", specificity))
    else:
        checklist.append(("Risk Avoidance (Specificity >= 70%)", "✗ FAIL", specificity))
    
    # Sample Size
    if len(y_test) >= 500:
        checklist.append(("Sufficient Test Sample (N >= 500)", "✓ PASS", len(y_test)))
    else:
        checklist.append(("Sufficient Test Sample (N >= 500)", "✗ FAIL", len(y_test)))
    
    print(f"\n{'Criterion':<45} {'Status':<12} {'Value':<10}")
    print("-" * 70)
    for criterion, status, value in checklist:
        print(f"{criterion:<45} {status:<12} {value:.4f}")
    
    passing = sum(1 for _, status, _ in checklist if "✓" in status)
    total = len(checklist)
    
    print("-" * 70)
    print(f"Score: {passing}/{total} criteria met\n")
    
    if passing == total:
        print("VERDICT: ✓ APPROVED FOR PRODUCTION")
        print("The model meets all minimum credit risk standards and is suitable for lending decisions.")
    elif passing >= total - 1:
        print("VERDICT: ⚠ CONDITIONAL APPROVAL")
        print("The model needs remediation on 1 criterion before production deployment.")
    elif passing >= total - 2:
        print("VERDICT: ✗ NOT APPROVED")
        print("The model requires improvements on 2+ criteria. Additional feature engineering or")
        print("model refinement recommended before considering for production use.")
    else:
        print("VERDICT: ✗ STRONGLY NOT RECOMMENDED")
        print("The model's predictive power is insufficient for credit risk decisions.")
        print("Recommend: Complete redesign with different features or data sources.")
    
    print("\n" + "=" * 80)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
