# RFECV Validation Execution Guide
## Complete Instructions for Running Full Feature Selection Validation

---

## Overview

This guide provides step-by-step instructions for executing the comprehensive RFECV validation study that compares IV-based and RFECV feature selection approaches for the Restaurant PD Model.

**Status**: ✓ Notebook created and committed to GitHub  
**Runtime**: ~15-20 minutes  
**Outputs**: Full comparison report with visualizations

---

## What's Being Executed

### Notebook: `Restaurant_PD_Model_RFECV_Validation.ipynb`

**14 Comprehensive Sections**:

1. **Setup & Dependencies**: Load all required libraries
2. **Data Loading**: Load 3.5M transactions using Polars
3. **Restaurant-Level Aggregation**: Aggregate to 10,812 restaurants
4. **Feature Preparation**: 74+ features after encoding
5. **Train-Test Split**: 80/20 stratified split (8,649 train / 2,163 test)
6. **IV-Based Selection**: OptBinning WoE/IV analysis (expected 28-32 features)
7. **WoE Encoding**: Transform features using Weight of Evidence
8. **RFECV**: 5-fold stratified cross-validation (10-15 min runtime)
9. **Feature Overlap**: Analyze selection differences
10. **Model Training**: Train EBT with both feature sets
11. **Performance Comparison**: Side-by-side metrics (AUC, K-S, calibration, sensitivity, specificity)
12. **Calibration Analysis**: Decile-by-decile comparison
13. **Visualization**: 9-panel comprehensive comparison charts
14. **Final Report**: Production recommendation

---

## How to Execute

### Option 1: Jupyter Notebook (Recommended)

```bash
# Navigate to project directory
cd "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT"

# Open in Jupyter
jupyter notebook Restaurant_PD_Model_RFECV_Validation.ipynb

# Execute all cells: Cell → Run All Cells
# Or press: Ctrl+Shift+Enter (all cells) or Shift+Enter (individual cells)
```

**Expected Runtime**: 15-20 minutes

### Option 2: Command Line Execution

```bash
cd "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT"

# Execute and save results
jupyter nbconvert --to notebook --execute Restaurant_PD_Model_RFECV_Validation.ipynb \
  --ExecutePreprocessor.timeout=1800 \
  --output Restaurant_PD_Model_RFECV_Validation_RESULTS.ipynb
```

### Option 3: Background Execution

```bash
# Run in background and capture output
timeout 1800 python -m jupyter nbconvert --to notebook --execute \
  Restaurant_PD_Model_RFECV_Validation.ipynb \
  --ExecutePreprocessor.timeout=1800 > execution.log 2>&1 &

# Monitor progress
tail -f execution.log
```

---

## Expected Outputs

### 1. Feature Selection Results

**IV-Based Selection**:
- Selected Features: 28-32 (IV >= 0.02, < 1.0)
- Method: OptBinning Weight of Evidence
- IV Range: 0.0200 - 0.5000

**RFECV Selection**:
- Selected Features: 20-25 (70-85% overlap with IV)
- Method: EBT with 5-fold cross-validation
- Feature Reduction: 15-35% fewer features

**Feature Overlap**:
- Expected Agreement: 70-85%
- Features Only in IV-Based: 5-8
- Features Only in RFECV: 2-4

### 2. Performance Metrics Comparison

| Metric | IV-Based | RFECV | Expected Difference |
|--------|----------|-------|-------------------|
| **Test AUC** | 0.7880 | 0.78-0.80 | ±1% |
| **K-S Statistic** | 0.4542 | 0.45-0.46 | Negligible |
| **Gini Coefficient** | 0.5761 | 0.56-0.58 | < 2% |
| **Sensitivity** | 44.88% | 44-48% | ±3% |
| **Specificity** | 92.34% | 92-94% | ±2% |
| **Precision** | ~40% | 40-42% | ±2% |
| **Calibration MAE** | 0.0180 | 0.017-0.019 | Similar |

### 3. Visualizations Generated

**9-Panel Comprehensive Comparison**:
1. AUC Comparison (Train vs Test)
2. K-S & Gini Metrics
3. Feature Count Comparison
4. ROC Curves Overlaid
5. Sensitivity vs Specificity
6. Calibration Error (MAE)
7. Calibration Curves by Decile
8. Confusion Matrix (IV-Based)
9. Confusion Matrix (RFECV)

**Additional Plots**:
- RFECV Elbow Curve (optimal feature count)
- Feature Importance Comparison (top 15 features per model)

### 4. Final Report

**Key Findings**:
- ✓ Both approaches show similar test AUC (difference < 1%)
- ✓ High feature overlap (70-85%) confirms robust selection
- ✓ Similar generalization gaps (3-5% overfitting)
- ✓ Excellent calibration for both approaches
- ✓ RFECV validates IV-based methodology

**Production Recommendation**:
- Proceed with **IV-Based features** (current production model)
- Rationale: Faster, more interpretable, industry standard
- RFECV provides independent cross-validation confirmation

---

## Interpreting Results

### Understanding the Comparison

1. **Feature Overlap (70-85% agreement)**
   - High overlap means both methods identify strong predictive signals
   - Confirms robustness of variable selection
   - Indicates no data leakage in excluded features

2. **AUC Parity (±1%)**
   - Similar test AUC validates feature selection quality
   - RFECV accounts for multivariate relationships
   - IV-Based is simpler and more transparent
   - Both suitable for production

3. **Calibration Analysis**
   - Expected vs Actual default rates align well
   - MAE < 0.02 indicates excellent calibration
   - Both models are well-calibrated across deciles
   - No systematic bias in predictions

4. **Generalization Gap**
   - 3-5% gap is acceptable (indicates good generalization)
   - No evidence of overfitting with either approach
   - Cross-validation maintains consistency

### What Good Results Look Like

✅ **Expected (Passing)**:
- Test AUC difference < 1% 
- Feature overlap > 70%
- Generalization gap < 5%
- Calibration MAE < 0.05
- Similar sensitivity/specificity

❌ **Red Flags (Failing)**:
- Test AUC difference > 2%
- Feature overlap < 50%
- Generalization gap > 10%
- High calibration error
- Very different sensitivity/specificity

---

## Troubleshooting

### Issue: Kernel Dies During RFECV

**Cause**: RFECV is memory-intensive with 5-fold CV

**Solutions**:
1. Close other applications to free RAM
2. Reduce number of folds: `StratifiedKFold(n_splits=3)`
3. Run on machine with ≥ 8GB RAM
4. Use `n_jobs=-1` for parallel processing

### Issue: Protobuf Warning Messages

**Cause**: Common with `interpret` library

**Solution**: Can be safely ignored - warnings don't affect execution

### Issue: Slow Execution

**Normal**: RFECV with EBT takes 10-15 minutes
- Progress visible in cell outputs
- Each fold takes 2-3 minutes
- 5 folds total = ~15 minutes

### Issue: Feature Selection Mismatch

**Possible Causes**:
- Different random_state values
- Changes to feature engineering
- Different IV thresholds

**Check**: Verify input data is identical before and after

---

## Next Steps After Execution

### 1. Review Results

```python
# Check test AUC values
print(f"IV-Based Test AUC: {auc_test_iv:.4f}")
print(f"RFECV Test AUC: {auc_test_rfecv:.4f}")

# Check feature overlap
overlap = set(selected_iv) & set(selected_rfecv)
print(f"Feature overlap: {len(overlap)}/{max(len(selected_iv), len(selected_rfecv))}")
```

### 2. Save Results

```python
# Save selected features
selected_iv_df = pd.DataFrame({'Feature': selected_iv, 'Method': 'IV-Based'})
selected_rfecv_df = pd.DataFrame({'Feature': selected_rfecv, 'Method': 'RFECV'})
pd.concat([selected_iv_df, selected_rfecv_df]).to_csv('feature_selections.csv', index=False)

# Save comparison metrics
comparison_df.to_csv('performance_comparison.csv', index=False)
```

### 3. Make Production Decision

**If IV-Based is still best**:
- Current model is validated
- Continue with production deployment
- RFECV provides backup methodology

**If RFECV performs better**:
- Consider switching to RFECV features
- May provide better AUC with fewer features
- Requires model retraining and testing

### 4. Document Findings

- Update model documentation
- Record feature selection methodology
- Document performance metrics
- Create comparison summary for stakeholders

---

## Quick Reference: Key Metrics

### Production Acceptance Criteria

| Criterion | Minimum | Target |
|-----------|---------|--------|
| Test AUC | 0.75 | > 0.78 |
| K-S Statistic | 0.30 | > 0.40 |
| Feature Overlap | 70% | > 75% |
| AUC Difference | < 2% | < 1% |
| Sensitivity | 40% | > 44% |
| Specificity | 70% | > 92% |
| Calibration MAE | < 0.10 | < 0.02 |

### Expected Times (Minutes)

| Step | Duration |
|------|----------|
| Data loading | 0.5-1 |
| IV analysis | 1-2 |
| RFECV (5-fold CV) | 10-15 |
| Model training | 2-3 |
| Evaluation | 1-2 |
| **Total** | **15-20** |

---

## Files Generated

### Input Files
- `Restaurant_PD_Model_RFECV_Validation.ipynb` (notebook)

### Output Files (Generated after execution)
- `Restaurant_PD_Model_RFECV_Validation.ipynb` (with results)
- Visualizations (embedded in notebook)
- Performance comparison (printed in notebook)

### Reference Files
- `RFECV_Validation_Summary.ipynb` (framework document)
- `run_pipeline_ebt_with_rfecv.py` (alternative pipeline)
- `feature_engineering_timeseries.py` (feature definitions)

---

## GitHub Repository

**Location**: https://github.com/pradark/Restaurant-PD-model.git  
**Branch**: main  
**Commit**: "Add comprehensive RFECV validation notebook"

**To View Results on GitHub**:
```bash
# Check notebook on GitHub
https://github.com/pradark/Restaurant-PD-model/blob/main/Principal%20Data%20Scientist%20Capital%20Case%20Study/polars_EBT/Restaurant_PD_Model_RFECV_Validation.ipynb
```

---

## Support & Questions

**If results show:**

1. **AUC difference > 2%**: Re-check feature encoding, ensure WoE is applied correctly
2. **Feature overlap < 50%**: Investigate data quality, consider feature engineering changes
3. **High calibration error**: Review binning strategy, check for class imbalance
4. **Memory issues**: Reduce CV folds or close other applications

**For questions**: Review RFECV_Validation_Summary.ipynb for comprehensive background

---

## Success Checklist

✅ Notebook created and committed  
✅ All 14 sections documented  
✅ Expected outputs defined  
✅ Execution guide provided  
✅ Troubleshooting tips included  
✅ Decision framework documented  
✅ Next steps outlined  

**Ready to execute**: Run the notebook using Jupyter and review the comprehensive comparison results!

---

**Last Updated**: 2026-05-13  
**Version**: 1.0 Complete  
**Status**: ✓ Ready for Execution
