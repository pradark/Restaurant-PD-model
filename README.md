# Restaurant PD Model

This repository contains restaurant probability-of-default modeling notebooks and supporting code for the Toast capital case study.

## Main Notebook

- `Restaurant_Default_Model_Executed.ipynb` is the final validated notebook with saved outputs.
- The notebook was updated for current package APIs and local data paths, then executed successfully end-to-end.

## Interactive Report

- `index.html` is a self-contained interactive report of the analysis (model performance, feature importance, decile risk, calibration, VIF, and a searchable holdout risk table). Open it directly in a browser, or view it via GitHub Pages once enabled for this repository.

## Environment

Install the package stack with:

```bash
python3 -m venv .venv_restaurant
.venv_restaurant/bin/python -m pip install -U pip setuptools wheel
.venv_restaurant/bin/python -m pip install -r requirements.txt
```

The `requirements.txt` file is pinned from the environment that successfully ran the notebook. For conda-compatible tooling, use:

```bash
conda env create -f environment.yml
```

Register the kernel:

```bash
.venv_restaurant/bin/python -m ipykernel install --user --name restaurant-venv --display-name "Python (restaurant venv)"
```

## Data

Large raw data files and generated artifacts are intentionally excluded from git. Place the lending default CSV files under:

```text
Principal Data Scientist Capital Case Study/
```

The notebook expects the standard files:

- `Lending_default_train_tx.csv`
- `Lending_default_train_account.csv`
- `Lending_default_train_label.csv`
- `Lending_default_holdout_tx.csv`
- `Lending_default_holdout_account.csv`
