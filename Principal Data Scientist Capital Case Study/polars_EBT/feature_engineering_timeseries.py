#!/usr/bin/env python3
"""
Time-Series Feature Engineering for PD Model
==============================================
Builds rolling window features from transaction data using Polars.

Features Generated:
- Rolling window aggregations (7d, 30d, 90d, 180d)
- Momentum features (percent change vs lag dates)
- Temporal features (day of week, season, etc.)
- Coefficient of variation (volatility measures)

Output:
- train_features_timeseries.csv
- holdout_features_timeseries.csv
- train_merged_complete.parquet
- holdout_merged_complete.parquet
"""

import os
import polars as pl
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

DATA_PATHS = {
    'train_tx': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_tx.csv",
    'train_account': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_account.csv",
    'train_label': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_train_label.csv",
    'holdout_tx': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_holdout_tx.csv",
    'holdout_account': "/Users/pradark/Documents/011. Work/Toast/Export/Lending_default_holdout_account.csv",
}

OUTPUT_DIR = "/Users/pradark/Documents/011. Work/Toast/Principal Data Scientist Capital Case Study/polars_EBT/feature_engineering_output"
LABEL_KEY = 'Restaurant_ID'

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("=" * 80)
print("TIME-SERIES FEATURE ENGINEERING PIPELINE")
print("=" * 80)
print("\n[STEP 1/7] Loading data...")

train_tx = pl.read_csv(DATA_PATHS['train_tx'])
train_account = pl.read_csv(DATA_PATHS['train_account'])
train_label = pl.read_csv(DATA_PATHS['train_label'])
holdout_tx = pl.read_csv(DATA_PATHS['holdout_tx'])
holdout_account = pl.read_csv(DATA_PATHS['holdout_account'])

print(f"  ✓ Training Transactions: {train_tx.shape[0]:,} rows × {train_tx.shape[1]} cols")
print(f"  ✓ Training Accounts: {train_account.shape[0]:,} rows × {train_account.shape[1]} cols")
print(f"  ✓ Training Labels: {train_label.shape[0]:,} rows × {train_label.shape[1]} cols")
print(f"  ✓ Holdout Transactions: {holdout_tx.shape[0]:,} rows × {holdout_tx.shape[1]} cols")
print(f"  ✓ Holdout Accounts: {holdout_account.shape[0]:,} rows × {holdout_account.shape[1]} cols")

# ============================================================================
# STEP 2: STANDARDIZE DATAFRAMES
# ============================================================================

print("\n[STEP 2/7] Standardizing dataframes...")

df_train_account = train_account.clone()
df_train_tx = train_tx.clone()
df_train_label = train_label.clone()
df_holdout_account = holdout_account.clone()
df_holdout_tx = holdout_tx.clone()

print(f"  ✓ All dataframes standardized with df_ prefix")

# ============================================================================
# STEP 3: DEFINE TIME-SERIES FEATURE ENGINEERING FUNCTION
# ============================================================================

print("\n[STEP 3/7] Defining feature engineering function...")


def build_timeseries_features(df_tx, windows=(7, 30, 90, 180)):
    """
    Build rolling time-series features from transaction data.
    
    Parameters:
    - df_tx: Polars DataFrame with transaction data
    - windows: Tuple of rolling window sizes (in days)
    
    Returns:
    - Polars DataFrame with rolling features and temporal features
    """
    
    # Prepare transaction data
    df_work = (
        df_tx
        .select([LABEL_KEY, 'Tx_date', 'processing_volume', 'Tx_hours'])
        .with_columns(pl.col('Tx_date').str.to_date(strict=False))
        .sort([LABEL_KEY, 'Tx_date'])
    )

    # Get unique snapshots
    df_snapshots = df_work.select([LABEL_KEY, 'Tx_date']).unique().sort([LABEL_KEY, 'Tx_date'])

    # Build rolling window aggregations
    for window in windows:
        df_window = (
            df_work
            .group_by_dynamic(
                index_column='Tx_date',
                group_by=LABEL_KEY,
                every='1d',
                period=f'{window}d',
                closed='right',
            )
            .agg([
                pl.col('processing_volume').mean().alias('avg_proc_vol'),
                pl.col('processing_volume').min().alias('min_proc_vol'),
                pl.col('processing_volume').max().alias('max_proc_vol'),
                pl.col('processing_volume').std().alias('std_proc_vol'),
                pl.col('Tx_hours').mean().alias('avg_tx_hours'),
                pl.col('Tx_hours').min().alias('min_tx_hours'),
                pl.col('Tx_hours').max().alias('max_tx_hours'),
                pl.col('Tx_hours').std().alias('std_tx_hours'),
            ])
            .with_columns([
                pl.when(pl.col('avg_proc_vol') != 0)
                .then(pl.col('std_proc_vol') / pl.col('avg_proc_vol'))
                .otherwise(None)
                .alias(f'cv_proc_vol_{window}d'),
                pl.when(pl.col('avg_tx_hours') != 0)
                .then(pl.col('std_tx_hours') / pl.col('avg_tx_hours'))
                .otherwise(None)
                .alias(f'cv_tx_hours_{window}d'),
            ])
            .rename({
                'avg_proc_vol': f'avg_proc_vol_{window}d',
                'min_proc_vol': f'min_proc_vol_{window}d',
                'max_proc_vol': f'max_proc_vol_{window}d',
                'avg_tx_hours': f'avg_tx_hours_{window}d',
                'min_tx_hours': f'min_tx_hours_{window}d',
                'max_tx_hours': f'max_tx_hours_{window}d',
            })
            .select([
                LABEL_KEY,
                'Tx_date',
                f'avg_proc_vol_{window}d',
                f'min_proc_vol_{window}d',
                f'max_proc_vol_{window}d',
                f'avg_tx_hours_{window}d',
                f'min_tx_hours_{window}d',
                f'max_tx_hours_{window}d',
                f'cv_proc_vol_{window}d',
                f'cv_tx_hours_{window}d',
            ])
        )

        df_snapshots = df_snapshots.join(df_window, on=[LABEL_KEY, 'Tx_date'], how='left')

    # Add momentum features
    df_daily = (
        df_work
        .select([LABEL_KEY, 'Tx_date', 'processing_volume', 'Tx_hours'])
        .unique()
        .rename({
            'processing_volume': 'curr_processing_volume',
            'Tx_hours': 'curr_tx_hours',
        })
    )
    df_snapshots = df_snapshots.join(df_daily, on=[LABEL_KEY, 'Tx_date'], how='left')

    for lag_days in windows:
        lag_df = (
            df_daily
            .with_columns((pl.col('Tx_date') + pl.duration(days=lag_days)).alias('Tx_date'))
            .rename({
                'curr_processing_volume': f'proc_vol_{lag_days}d_ago',
                'curr_tx_hours': f'tx_hours_{lag_days}d_ago',
            })
        )

        df_snapshots = df_snapshots.join(lag_df, on=[LABEL_KEY, 'Tx_date'], how='left')

        df_snapshots = df_snapshots.with_columns([
            pl.when(pl.col(f'proc_vol_{lag_days}d_ago') != 0)
            .then(
                (pl.col('curr_processing_volume') - pl.col(f'proc_vol_{lag_days}d_ago'))
                / pl.col(f'proc_vol_{lag_days}d_ago')
            )
            .otherwise(None)
            .alias(f'pct_change_proc_vol_vs_{lag_days}d_ago'),
            pl.when(pl.col(f'tx_hours_{lag_days}d_ago') != 0)
            .then(
                (pl.col('curr_tx_hours') - pl.col(f'tx_hours_{lag_days}d_ago'))
                / pl.col(f'tx_hours_{lag_days}d_ago')
            )
            .otherwise(None)
            .alias(f'pct_change_tx_hours_vs_{lag_days}d_ago'),
        ])

    # Add temporal features
    df_snapshots = df_snapshots.with_columns([
        pl.col('Tx_date').dt.weekday().alias('snapshot_day_of_week'),
        pl.col('Tx_date').dt.ordinal_day().alias('snapshot_day_of_year'),
        pl.col('Tx_date').dt.month().alias('snapshot_month'),
        pl.col('Tx_date').dt.quarter().alias('snapshot_quarter'),
    ])

    return df_snapshots.sort([LABEL_KEY, 'Tx_date'])


print("  ✓ Function defined")

# ============================================================================
# STEP 4: BUILD TRAINING FEATURES
# ============================================================================

print("\n[STEP 4/7] Building training features...")
df_train_features = build_timeseries_features(df_train_tx)
print(f"  ✓ Training features: {df_train_features.shape[0]:,} rows × {df_train_features.shape[1]} cols")

# ============================================================================
# STEP 5: BUILD HOLDOUT FEATURES
# ============================================================================

print("\n[STEP 5/7] Building holdout features...")
df_holdout_features = build_timeseries_features(df_holdout_tx)
print(f"  ✓ Holdout features: {df_holdout_features.shape[0]:,} rows × {df_holdout_features.shape[1]} cols")

# ============================================================================
# STEP 6: CLEAN AND MERGE DATA
# ============================================================================

print("\n[STEP 6/7] Cleaning and merging data...")

# Clean unnamed columns
df_train_account_clean = df_train_account.select([c for c in df_train_account.columns if not c.startswith('Unnamed:')])
df_train_label_clean = df_train_label.select([c for c in df_train_label.columns if not c.startswith('Unnamed:')])
df_holdout_account_clean = df_holdout_account.select([c for c in df_holdout_account.columns if not c.startswith('Unnamed:')])

# Merge datasets
df_train_merged = (
    df_train_features
    .join(df_train_account_clean, on=LABEL_KEY, how='left')
    .join(df_train_label_clean, on=LABEL_KEY, how='left')
)

df_holdout_merged = df_holdout_features.join(df_holdout_account_clean, on=LABEL_KEY, how='left')

print(f"  ✓ Train merged: {df_train_merged.shape[0]:,} rows × {df_train_merged.shape[1]} cols")
print(f"  ✓ Holdout merged: {df_holdout_merged.shape[0]:,} rows × {df_holdout_merged.shape[1]} cols")
print(f"  ✓ No Unnamed columns in train: {not any(c.startswith('Unnamed:') for c in df_train_merged.columns)}")
print(f"  ✓ No Unnamed columns in holdout: {not any(c.startswith('Unnamed:') for c in df_holdout_merged.columns)}")

# ============================================================================
# STEP 7: EXPORT DATA
# ============================================================================

print("\n[STEP 7/7] Exporting processed datasets...")

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Export CSV
df_train_features.write_csv(f'{OUTPUT_DIR}/train_features_timeseries.csv')
print(f"  ✓ train_features_timeseries.csv")

df_holdout_features.write_csv(f'{OUTPUT_DIR}/holdout_features_timeseries.csv')
print(f"  ✓ holdout_features_timeseries.csv")

df_train_merged.write_csv(f'{OUTPUT_DIR}/train_merged_complete.csv')
print(f"  ✓ train_merged_complete.csv")

df_holdout_merged.write_csv(f'{OUTPUT_DIR}/holdout_merged_complete.csv')
print(f"  ✓ holdout_merged_complete.csv")

# Export Parquet
df_train_merged.write_parquet(f'{OUTPUT_DIR}/train_merged_complete.parquet')
print(f"  ✓ train_merged_complete.parquet")

df_holdout_merged.write_parquet(f'{OUTPUT_DIR}/holdout_merged_complete.parquet')
print(f"  ✓ holdout_merged_complete.parquet")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 80)

print(f"\nDataset Shapes:")
print(f"  Train Features: {df_train_features.shape}")
print(f"  Holdout Features: {df_holdout_features.shape}")
print(f"  Train Merged: {df_train_merged.shape}")
print(f"  Holdout Merged: {df_holdout_merged.shape}")

print(f"\nFeature Categories:")
rolling = len([c for c in df_train_merged.columns if any(f'{w}d' in c for w in [7, 30, 90, 180])])
momentum = len([c for c in df_train_merged.columns if 'pct_change' in c])
temporal = len([c for c in df_train_merged.columns if 'snapshot' in c])
print(f"  Rolling Windows: {rolling}")
print(f"  Momentum: {momentum}")
print(f"  Temporal: {temporal}")
print(f"  Total Features: {len(df_train_merged.columns) - 1}")

print(f"\nOutput Directory: {OUTPUT_DIR}")
print(f"\nNext Steps:")
print(f"  1. Load datasets from Parquet (more efficient)")
print(f"  2. Use for PD model training")
print(f"  3. Features are ready for modeling")

print("=" * 80)
