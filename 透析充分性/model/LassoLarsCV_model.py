# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn import linear_model
from sklearn.pipeline import make_pipeline  # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler  # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score  # 引入K折交叉验证


from sklearn.metrics import r2_score, mean_absolute_error, make_scorer


from yellowbrick.regressor import prediction_error

from sklearn.pipeline import Pipeline

import pickle


def LassoLarsCV_model(target, X_train, y_train, X_test, y_test):
    pipe_svr = Pipeline(
        [("StandardScaler", StandardScaler()), ("Lasso", linear_model.LassoLarsCV())]
    )
    param_range = range(1, 1000)
    param_dist = [
        {'Lasso__max_iter': param_range},
        # 'random_state': [0],
        # 'alpha': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1],
    ]

    gs = GridSearchCV(
        estimator=pipe_svr, param_grid=param_dist, scoring='r2', cv=2, n_jobs=-1
    )

    gs = gs.fit(X_train, y_train)
    print("网格搜索最优得分：", gs.best_score_)
    print("网格搜索最优参数组合：\n", gs.best_params_)

    model = gs.best_estimator_

    print('Total training', model.score(X_train, y_train))
    print('Total testing', model.score(X_test, y_test))
    print('Total testing', mean_absolute_error(y_test, model.predict(X_test)))

    visualizer = prediction_error(model, X_train, y_train, X_test, y_test)

    with open('LassoLarsCV/' + str(target) + ' model.pickle', 'wb') as f:
        pickle.dump(model, f)

    return [
        model.score(X_test, y_test),
        mean_absolute_error(y_test, model.predict(X_test)),
    ]
