# -*- coding: utf-8 -*-

import numpy as np
import xgboost
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr

def spearman_score(x, y):
    return spearmanr(x, y)[0]

def test_CV2():

    def func(X):
        return ((X[:,0]+0.1)*(X[:,1]-2.2))**2

    np.random.seed(101)
    X_train = np.random.random((1000, 2))
    y_train = func(X_train)
    # print(y_train)

    params = {'learning_rate':[0.5, 0.2, 0.1, 0.05, 0.02, 0.01],
              'gamma':[0.1, 0.2, 0.5, 1, 2, 5],
              'reg_alpha':10. * np.arange(-8, 2, .25),
              'reg_lambda':10. * np.arange(-8, 2, .25),
              'subsample': [0.1, 0.2, 0.5, 0.7, 0.9],
              'max_depth': [1, 2, 3]
              }
    model = RandomizedSearchCV(xgboost.XGBRegressor(), param_distributions=params, n_iter=100,
                               scoring=make_scorer(spearman_score), cv=5, n_jobs=-1, verbose=1, random_state=1001)

    model.fit(X_train, y_train)

def test_CV():
    import numpy as np

    from time import time
    from scipy.stats import randint as sp_randint

    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier

    # get some data
    digits = load_digits()
    X, y = digits.data, digits.target

    # build a classifier
    clf = RandomForestClassifier(n_estimators=20)


    # Utility function to report best scores
    def report(results, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                      results['mean_test_score'][candidate],
                      results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")


    # specify parameters and distributions to sample from
    param_dist = {"max_depth": [3, None],
                  "max_features": sp_randint(1, 11),
                  "min_samples_split": sp_randint(2, 11),
                  "min_samples_leaf": sp_randint(1, 11),
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       n_iter=n_iter_search)

    start = time()
    random_search.fit(X, y)
    print("RandomizedSearchCV took %.2f seconds for %d candidates"
          " parameter settings." % ((time() - start), n_iter_search))
    report(random_search.cv_results_)

    # use a full grid over all parameters
    param_grid = {"max_depth": [3, None],
                  "max_features": [1, 3, 10],
                  "min_samples_split": [2, 3, 10],
                  "min_samples_leaf": [1, 3, 10],
                  "bootstrap": [True, False],
                  "criterion": ["gini", "entropy"]}

    # run grid search
    grid_search = GridSearchCV(clf, param_grid=param_grid)
    start = time()
    grid_search.fit(X, y)

    print("GridSearchCV took %.2f seconds for %d candidate parameter settings."
          % (time() - start, len(grid_search.cv_results_['params'])))
    report(grid_search.cv_results_)
