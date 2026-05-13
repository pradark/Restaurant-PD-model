import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
import lightgbm as lgb


def bayes_parameter_opt_lgb(X, y, categorical_feats, init_round=15, opt_round=25, n_folds=3, random_seed=137, n_estimators=1000, output_process=False,monotonic_contraints=None):
    import warnings
    warnings.filterwarnings("ignore")
    # prepare data
    train_data = lgb.Dataset(data=X, label=y, categorical_feature = categorical_feats, free_raw_data=False)
    # parameters
    def lgb_eval(num_leaves, min_data_in_leaf, learning_rate, min_sum_hessian_in_leaf, feature_fraction, bagging_fraction, max_depth, lambda_l1, lambda_l2, min_split_gain, min_child_weight):
        
        params = {'objective':'binary', 'num_iterations': n_estimators, 'learning_rate':learning_rate,
        'boost_from_average': False, 'is_unbalance': False, 'early_stopping_round':100, 'verbose':-1, 'metric':'auc', 'monotone_constraints' : monotonic_contraints}
        
        params["num_leaves"] = int(round(num_leaves))
        params['feature_fraction'] = max(min(feature_fraction, 1), 0)
        params['bagging_fraction'] = max(min(bagging_fraction, 1), 0)
        params['max_depth'] = int(round(max_depth))
        params['lambda_l1'] = max(lambda_l1, 0)
        params['lambda_l2'] = max(lambda_l2, 0)
        params['min_split_gain'] = min_split_gain
        params['min_child_weight'] = min_child_weight
        #cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, verbose_eval=200, metrics=['auc'])
        cv_result = lgb.cv(params, train_data, nfold=n_folds, seed=random_seed, stratified=True, metrics=['auc'])
        # print(cv_result)
        return max(cv_result['valid auc-mean'])
        # return 1
 


    # Bounded region of parameter space
    bounds_LGB = {
        'num_leaves': (15, 45), 
        'min_data_in_leaf': (5, 20),  
        'learning_rate': (0.01, 0.3),
        'min_sum_hessian_in_leaf': (0.00001, 0.01),    
        'feature_fraction': (0.05, 0.9),
        'bagging_fraction': (0.8, 1),
        'max_depth':(5,12),
        'lambda_l1': (0, 5.0), 
        'lambda_l2': (0, 5.0), 
        'min_split_gain': (0.001, 0.5),              
        'min_child_weight': (5, 50),
    }
    
    lgbBO = BayesianOptimization(lgb_eval, bounds_LGB, random_state=42)
    # optimize
    lgbBO.maximize(init_points=init_round, n_iter=opt_round)
    
    # output optimization process
    if output_process==True: lgbBO.points_to_csv("bayes_opt_result.csv")
    
    # return best parameters
    return lgbBO.max['target'], lgbBO.max['params']

# Function to split long description into smaller chunks in multiple lines
def split_text_newline(text,span=15):
    words = text.split(' ')
    words_list = [' '.join(words[i:i+span]) for i in range(0, len(words), span)]
    text = '\n '.join(words_list)
    return text