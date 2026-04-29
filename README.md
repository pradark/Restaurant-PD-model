# Toast Probability of Default Modeling

This repository contains restaurant probability-of-default modeling notebooks and supporting code for the Toast capital case study.

## Main Notebook

- `Restaurant_Default_Model-Copy1.ipynb` is the validated notebook run in the local `restaurant-venv` kernel.
- The notebook was updated for current package APIs and local data paths, then executed successfully end-to-end.

## Environment

Install the package stack with:

```bash
python3 -m venv .venv_restaurant
.venv_restaurant/bin/python -m pip install -U pip setuptools wheel
.venv_restaurant/bin/python -m pip install -r requirements.txt
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
