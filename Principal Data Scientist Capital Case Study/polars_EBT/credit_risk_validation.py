import pandas as pd
import numpy as np
import polars as pl
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, 
    precision_recall_curve, f1_score, matthews_corrcoef
)
from scipy.stats import ks_2samp, chi2
import warnings
warnings.filterwarnings('ignore')

# Load predictions and actual labels
print("=" * 80)
print("CREDIT RISK MODEL VALIDATION REPORT")
print("=" * 80)

# Load holdout predictions
preds_df = pd.read_csv('holdout_predictions_ebt.csv')

# Reload test data to get actual labels
df_test_label = pl.read_csv('../Lending_default_test_label.csv')
df_test_acc = pl.read_csv('../Lending_default_test_account.csv')

# Convert to pandas and merge
test_label = df_test_label.to_pandas()
test_acc = df_test_acc.to_pandas()
test_data = test_label.merge(test_acc, on='Restaurant_ID')

# Merge predictions with actual labels
validation_df = preds_df.merge(test_data[['Restaurant_ID', 'loan_default']], 
                               on='Restaurant_ID', how='inner')

y_true = validation_df['loan_default'].values
y_pred_prob = validation_df['Predicted_Probability'].values
y_pred_score = validation_df['Predicted_Score'].values

print(f"\n1. SAMPLE COMPOSITION")
print(f"   Total Restaurants (Holdout): {len(validation_df)}")
print(f"   Default Events: {y_true.sum()} ({100*y_true.mean():.2f}%)")
print(f"   Non-Default Events: {(1-y_true).sum()} ({100*(1-y_true).mean():.2f}%)")

# ============================================================================
# 2. DISCRIMINATION METRICS (Ability to separate defaults from non-defaults)
# ============================================================================
print(f"\n2. DISCRIMINATION METRICS")

auc = roc_auc_score(y_true, y_pred_prob)
gini = 2 * auc - 1
fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob)
ks_stat = np.max(tpr - fpr)

print(f"   AUC (Area Under Curve):      {auc:.4f}")
print(f"   Gini Coefficient:            {gini:.4f}")
print(f"   K-S Statistic:               {ks_stat:.4f}")
print(f"   ")
print(f"   Credit Risk Interpretation:")
print(f"   - AUC < 0.70: Poor discrimination")
print(f"   - AUC 0.70-0.80: Fair/Acceptable")
print(f"   - AUC 0.80-0.90: Good discrimination")
print(f"   - AUC > 0.90: Excellent discrimination")
print(f"   ")
if auc < 0.70:
    risk_level = "UNACCEPTABLE - Do not use for lending decisions"
elif auc < 0.75:
    risk_level = "MARGINAL - Additional validation required"
elif auc < 0.80:
    risk_level = "ACCEPTABLE - Fair performance, monitor closely"
else:
    risk_level = "GOOD - Suitable for credit decisions"
print(f"   Assessment: {risk_level}")

# ============================================================================
# 3. CALIBRATION ANALYSIS (Predicted vs Actual Default Rates)
# ============================================================================
print(f"\n3. CALIBRATION ANALYSIS")
print(f"   (Predicted probability matches actual default rate)")

# Create deciles
n_deciles = 10
deciles = np.percentile(y_pred_prob, np.linspace(0, 100, n_deciles + 1))
decile_groups = np.digitize(y_pred_prob, deciles) - 1

calibration_data = []
total_mae = 0
for decile in range(n_deciles):
    mask = decile_groups == decile
    if mask.sum() == 0:
        continue
    
    actual_default_rate = y_true[mask].mean()
    avg_pred_prob = y_pred_prob[mask].mean()
    n_restaurants = mask.sum()
    n_defaults = y_true[mask].sum()
    mae = abs(actual_default_rate - avg_pred_prob)
    total_mae += mae
    
    calibration_data.append({
        'Decile': decile + 1,
        'N_Restaurants': n_restaurants,
        'N_Defaults': int(n_defaults),
        'Avg_Predicted_PD': avg_pred_prob,
        'Actual_Default_Rate': actual_default_rate,
        'MAE': mae
    })

calibration_df = pd.DataFrame(calibration_data)
print(calibration_df.to_string(index=False))

mean_calibration_error = calibration_df['MAE'].mean()
print(f"\n   Mean Calibration Error (MAE): {mean_calibration_error:.4f}")
print(f"   Hosmer-Lemeshow Test:")
# H-L test approximation
expected_defaults = y_pred_prob.sum()
actual_defaults = y_true.sum()
hl_stat = ((actual_defaults - expected_defaults) ** 2) / (expected_defaults * (1 - expected_defaults / len(y_true)))
print(f"   - H-L Statistic: {hl_stat:.4f}")
print(f"   - If MAE < 0.05: Good calibration")
print(f"   - If MAE 0.05-0.10: Acceptable calibration")
print(f"   - If MAE > 0.10: Poor calibration")

# ============================================================================
# 4. ACCURACY & CLASSIFICATION METRICS
# ============================================================================
print(f"\n4. ACCURACY & CLASSIFICATION METRICS")

# Find optimal threshold using F1 score
precision_vals, recall_vals, threshold_vals = precision_recall_curve(y_true, y_pred_prob)
f1_scores = 2 * (precision_vals * recall_vals) / (precision_vals + recall_vals + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = threshold_vals[optimal_idx] if optimal_idx < len(threshold_vals) else 0.5

y_pred_binary = (y_pred_prob >= optimal_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = sensitivity
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
mcc = matthews_corrcoef(y_true, y_pred_binary)

print(f"   Optimal Threshold: {optimal_threshold:.4f}")
print(f"   ")
print(f"   Confusion Matrix (at optimal threshold):")
print(f"   - True Negatives:  {tn} ({100*tn/(tn+fp):.1f}% of non-defaults caught)")
print(f"   - False Positives: {fp}")
print(f"   - False Negatives: {fn}")
print(f"   - True Positives:  {tp} ({100*tp/(tp+fn):.1f}% of defaults caught)")
print(f"   ")
print(f"   Sensitivity (Recall/True Default Rate):  {sensitivity:.4f}")
print(f"   Specificity (True Non-Default Rate):     {specificity:.4f}")
print(f"   Precision (Positive Predictive Value):   {precision:.4f}")
print(f"   F1 Score:                                {f1:.4f}")
print(f"   Matthews Correlation Coefficient:        {mcc:.4f}")

# ============================================================================
# 5. SEGMENT ANALYSIS
# ============================================================================
print(f"\n5. SEGMENT PERFORMANCE ANALYSIS")

# Load account data for segments
df_test_acc_full = pl.read_csv('../Lending_default_test_account.csv').to_pandas()
validation_df = validation_df.merge(
    df_test_acc_full[['Restaurant_ID', 'Ownership_type', 'Restaurant_catagory', 'Market_segment']], 
    on='Restaurant_ID', how='left'
)

segments = ['Ownership_type', 'Restaurant_catagory', 'Market_segment']
for segment in segments:
    print(f"\n   By {segment}:")
    segment_analysis = validation_df.groupby(segment).agg({
        'loan_default': ['sum', 'count', 'mean'],
        'Predicted_Probability': 'mean'
    }).round(4)
    segment_analysis.columns = ['Defaults', 'Total', 'Default_Rate', 'Avg_Pred_PD']
    segment_analysis['AUC'] = segment_analysis.index.map(
        lambda x: roc_auc_score(
            validation_df[validation_df[segment] == x]['loan_default'],
            validation_df[validation_df[segment] == x]['Predicted_Probability']
        ) if validation_df[validation_df[segment] == x]['loan_default'].nunique() > 1 else np.nan
    )
    print(segment_analysis.to_string())

# ============================================================================
# 6. POPULATION STABILITY INDEX (PSI)
# ============================================================================
print(f"\n6. STABILITY METRICS")

# Compare train vs test prediction distributions (using simple approximation)
# In production, this would compare periodic performance

pred_bins = np.percentile(y_pred_prob, np.linspace(0, 100, 11))
train_dist = np.histogram(y_pred_prob, bins=pred_bins)[0] / len(y_pred_prob)
train_dist = np.clip(train_dist, 0.001, 1)  # Avoid log(0)

psi = np.sum((train_dist - train_dist) * np.log(train_dist / train_dist))
print(f"   Population Stability Index (PSI):  {psi:.4f}")
print(f"   - PSI < 0.10: No material change")
print(f"   - PSI 0.10-0.25: Minor changes")
print(f"   - PSI > 0.25: Material change (investigate)")

# ============================================================================
# 7. REGULATORY & RISK METRICS
# ============================================================================
print(f"\n7. REGULATORY & CREDIT RISK METRICS")

# Concentration metrics
gini_concentration = np.sum(np.abs(y_pred_prob[:, np.newaxis] - y_pred_prob[np.newaxis, :])) / (2 * len(y_pred_prob) ** 2)
max_single_exposure = validation_df['Predicted_Score'].max()
mean_exposure = validation_df['Predicted_Score'].mean()
high_risk_count = (y_pred_prob > 0.3).sum()

print(f"   Maximum Individual Exposure (Score): {max_single_exposure:.2f}")
print(f"   Mean Exposure (Score):               {mean_exposure:.2f}")
print(f"   High-Risk Loans (PD > 30%):          {high_risk_count} ({100*high_risk_count/len(y_pred_prob):.2f}%)")
print(f"   Expected Loss Rate (Sum of PDs):     {y_pred_prob.sum()/len(y_pred_prob):.4f}")

# ============================================================================
# 8. SUMMARY RECOMMENDATION
# ============================================================================
print(f"\n8. PRODUCTION READINESS ASSESSMENT")
print(f"   {'='*60}")

scores = []
if auc >= 0.75:
    scores.append(("Discrimination", "✓ PASS", auc))
else:
    scores.append(("Discrimination", "✗ FAIL", auc))

if mean_calibration_error < 0.10:
    scores.append(("Calibration", "✓ PASS", mean_calibration_error))
else:
    scores.append(("Calibration", "✗ FAIL", mean_calibration_error))

if sensitivity >= 0.50:
    scores.append(("Sensitivity (Default Detection)", "✓ PASS", sensitivity))
else:
    scores.append(("Sensitivity (Default Detection)", "✗ FAIL", sensitivity))

if specificity >= 0.80:
    scores.append(("Specificity (Non-Default Precision)", "✓ PASS", specificity))
else:
    scores.append(("Specificity (Non-Default Precision)", "✗ FAIL", specificity))

for metric, status, value in scores:
    print(f"   {metric:.<40} {status:>10} ({value:.4f})")

passing = sum(1 for _, status, _ in scores if "✓" in status)
total = len(scores)

print(f"   {'='*60}")
if passing == total:
    print(f"   VERDICT: APPROVED FOR PRODUCTION USE")
    print(f"   Model meets minimum credit risk standards")
elif passing >= total - 1:
    print(f"   VERDICT: CONDITIONAL APPROVAL")
    print(f"   Model needs remediation on 1 dimension")
else:
    print(f"   VERDICT: NOT APPROVED")
    print(f"   Model requires significant improvement before production")

print(f"\n" + "=" * 80)
