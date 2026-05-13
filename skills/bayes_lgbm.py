"""
bayes_lgbm.py — LightGBM with Bayesian hyperparameter optimisation
===================================================================
Anti-overfitting penalty built in: penalises train/val AUC gap > threshold.

Usage
-----
    from skills.bayes_lgbm import BayesLGBM

    blgbm = BayesLGBM(gap_threshold=0.02, gap_penalty=5.0,
                       n_init=10, n_iter=40, cv=5)
    blgbm.fit(X_train, y_train)

    print(blgbm.best_params_)
    proba = blgbm.predict_proba(X_test)[:, 1]
"""

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
import lightgbm as lgb


PBOUNDS = {
    "n_estimators":      (200, 600),
    "learning_rate":     (0.005, 0.05),
    "max_depth":         (3, 6),
    "num_leaves":        (10, 50),
    "min_child_samples": (30, 150),
    "reg_alpha":         (0.0, 2.0),
    "reg_lambda":        (0.0, 2.0),
    "colsample_bytree":  (0.4, 0.9),
    "subsample":         (0.6, 0.95),
}


class BayesLGBM:
    """
    LightGBM classifier with Bayesian hyperparameter search.

    Anti-overfitting objective:
        score = val_AUC - gap_penalty * max(0, gap - gap_threshold)

    Parameters
    ----------
    gap_threshold : float
        Maximum acceptable train/val AUC gap (default 0.02 = 2pp).
    gap_penalty : float
        Penalty multiplier for exceeding gap_threshold (default 5).
    n_init : int
        Random exploration steps before Bayesian exploitation.
    n_iter : int
        Bayesian optimisation iterations after init.
    cv : int
        Stratified K-fold folds.
    pbounds : dict or None
        Custom hyperparameter search bounds. Uses defaults if None.
    random_state : int
    """

    def __init__(self, gap_threshold: float = 0.02, gap_penalty: float = 5.0,
                 n_init: int = 10, n_iter: int = 40, cv: int = 5,
                 pbounds: dict = None, random_state: int = 42):
        self.gap_threshold = gap_threshold
        self.gap_penalty   = gap_penalty
        self.n_init        = n_init
        self.n_iter        = n_iter
        self.cv            = cv
        self.pbounds       = pbounds or PBOUNDS
        self.random_state  = random_state
        self.best_params_  = None
        self.model_        = None

    def _objective(self, X, y):
        def _fn(n_estimators, learning_rate, max_depth, num_leaves,
                min_child_samples, reg_alpha, reg_lambda,
                colsample_bytree, subsample):
            params = dict(
                n_estimators=     int(n_estimators),
                learning_rate=    learning_rate,
                max_depth=        int(max_depth),
                num_leaves=       int(num_leaves),
                min_child_samples=int(min_child_samples),
                reg_alpha=        reg_alpha,
                reg_lambda=       reg_lambda,
                colsample_bytree= colsample_bytree,
                subsample=        subsample,
                subsample_freq=   1,
                random_state=     self.random_state,
                n_jobs=           -1,
                verbose=          -1,
            )
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True,
                                  random_state=self.random_state)
            tr_aucs, val_aucs = [], []
            for tr_idx, val_idx in skf.split(X, y):
                Xf, Xv = X.iloc[tr_idx], X.iloc[val_idx]
                yf, yv = y.iloc[tr_idx], y.iloc[val_idx]
                m = lgb.LGBMClassifier(**params)
                m.fit(Xf, yf)
                tr_aucs.append( roc_auc_score(yf, m.predict_proba(Xf)[:, 1]))
                val_aucs.append(roc_auc_score(yv, m.predict_proba(Xv)[:, 1]))
            gap     = np.mean(tr_aucs) - np.mean(val_aucs)
            penalty = self.gap_penalty * max(0, gap - self.gap_threshold)
            return np.mean(val_aucs) - penalty
        return _fn

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: int = 0):
        """Run Bayesian search then train final model on full X, y."""
        optimizer = BayesianOptimization(
            f=            self._objective(X, y),
            pbounds=      self.pbounds,
            random_state= self.random_state,
            verbose=      verbose,
        )
        optimizer.maximize(init_points=self.n_init, n_iter=self.n_iter)

        raw = optimizer.max["params"]
        self.best_params_ = dict(
            n_estimators=     int(raw["n_estimators"]),
            learning_rate=    raw["learning_rate"],
            max_depth=        int(raw["max_depth"]),
            num_leaves=       int(raw["num_leaves"]),
            min_child_samples=int(raw["min_child_samples"]),
            reg_alpha=        raw["reg_alpha"],
            reg_lambda=       raw["reg_lambda"],
            colsample_bytree= raw["colsample_bytree"],
            subsample=        raw["subsample"],
            subsample_freq=   1,
        )

        self.model_ = lgb.LGBMClassifier(
            **self.best_params_,
            random_state=self.random_state,
            n_jobs=-1, verbose=-1,
        )
        self.model_.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_.predict_proba(X)

    def score_gap(self, X_tr, y_tr, X_te, y_te) -> dict:
        """Return AUC and KS for train and test, plus gaps."""
        tr_p = self.predict_proba(X_tr)[:, 1]
        te_p = self.predict_proba(X_te)[:, 1]
        return {
            "tr_auc":  roc_auc_score(y_tr, tr_p),
            "te_auc":  roc_auc_score(y_te, te_p),
            "tr_ks":   ks_2samp(tr_p[y_tr == 1], tr_p[y_tr == 0]).statistic,
            "te_ks":   ks_2samp(te_p[y_te == 1], te_p[y_te == 0]).statistic,
            "auc_gap": roc_auc_score(y_tr, tr_p) - roc_auc_score(y_te, te_p),
        }
