# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
    mean_squared_error,
)
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_class_weight
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from yellowbrick.regressor import prediction_error
from sklearn.metrics import (
    f1_score,
    precision_score,
    make_scorer,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)

import pickle


def XGB_model(target, X_train, y_train, X_test, y_test):
    # sourcery skip: extract-duplicate-method

    model = XGBRegressor()

    param_dist = {
        'booster': [
            'gbtree',
            'dart',
            'gblinear',
        ],
        'max_depth': [5, 10, 50, 100, 200, 500],
        'random_state': [0],
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
        'max_leaves': [2, 5, 10, 50, 100, 200, 500],
        'n_estimators': [50, 100, 200, 500],
        # 'class_weight': ['balanced', None],
    }
    scoring = {
        'r2': make_scorer(r2_score),
        # 'neg_mean_absolute_error': make_scorer(mean_absolute_error),
        'mean_squared_error': make_scorer(mean_squared_error),
    }

    grid_search = GridSearchCV(
        model,
        n_jobs=-1,
        param_grid=param_dist,
        cv=2,
        scoring=scoring,
        verbose=False,
        refit='r2',
    )

    grid_search.fit(X_train, y_train)

    print(grid_search.best_estimator_, ':best estimator of LGBM')

    model = grid_search.best_estimator_

    print('Total training', model.score(X_train, y_train))
    print('Total testing', model.score(X_test, y_test))

    visualizer = prediction_error(model, X_train, y_train, X_test, y_test)

    print('Total training', model.score(X_train, y_train))
    print('Total testing', model.score(X_test, y_test))

    y_pred = model.predict(X_test)

    x_ax = range(len(y_test))
    plt.figure(figsize=(12, 6))
    plt.plot(x_ax, y_test, label="original")
    plt.plot(x_ax, y_pred, label="predicted")
    plt.title("Boston dataset test and predicted data")
    plt.xlabel('X')
    plt.ylabel('Price')
    plt.legend(loc='best', fancybox=True, shadow=True)
    plt.grid(True)
    plt.show()

    with open('LGBM/' + str(target) + ' model.pickle', 'wb') as f:
        pickle.dump(model, f)

    return [
        model.score(X_test, y_test),
        mean_absolute_error(y_test, model.predict(X_test)),
    ]
