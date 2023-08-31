# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import pickle
from sklearn.model_selection import GridSearchCV

from sklearn.neural_network import MLPRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
    mean_squared_error,
)

from yellowbrick.regressor import prediction_error

import pickle


def MLP_model(target, X_train, y_train, X_test, y_test):
    model = MLPRegressor(random_state=0)

    param_dist = {
        'activation': ['identity', 'relu', 'logistic', 'tanh'],
        'max_iter': [10, 50, 100, 200, 500, 1000],
        'alpha': 10.0 ** -np.arange(1, 5),
        'hidden_layer_sizes': [10, 50, 100, 200, 500, 1000,],
        'solver': ['sgd', 'adam', 'lbfgs'],
        'learning_rate': ['constant', 'invscaling'],
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
