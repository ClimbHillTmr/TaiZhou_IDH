# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

from sklearn.svm import SVR  # 引入SVR类
from sklearn.pipeline import make_pipeline  # 引入管道简化学习流程
from sklearn.preprocessing import StandardScaler  # 由于SVR基于距离计算，引入对数据进行标准化的类
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV  # 引入网格搜索调优
from sklearn.model_selection import cross_val_score  # 引入K折交叉验证


from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    make_scorer,
    mean_absolute_percentage_error,
)


from yellowbrick.regressor import prediction_error

from sklearn.pipeline import Pipeline

import pickle

import matplotlib.pyplot as plt


def SVR_model(target, X_train, y_train, X_test, y_test):
    pipe_svr = Pipeline([("StandardScaler", StandardScaler()), ("svr", SVR())])
    param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    param_grid = [
        {"svr__C": param_range, "svr__kernel": ["linear"]},
        # 注意__是指两个下划线，一个下划线会报错的
        {"svr__C": param_range, "svr__gamma": param_range, "svr__kernel": ["rbf"]},
    ]

    gs = GridSearchCV(
        estimator=pipe_svr, param_grid=param_grid, scoring='r2', cv=4, n_jobs=-1
    )

    gs = gs.fit(X_train, y_train)
    print("网格搜索最优得分：", gs.best_score_)
    print("网格搜索最优参数组合：\n", gs.best_params_)

    model = gs.best_estimator_

    print('Total training', model.score(X_train, y_train))
    print('Total testing', model.score(X_test, y_test))
    print(
        'Total testing mean_absolute_error',
        mean_absolute_error(y_test, model.predict(X_test)),
    )
    print('Total testing r2', r2_score(y_test, model.predict(X_test)))
    print(
        'Total testingmean_absolute_percentage_error',
        mean_absolute_percentage_error(y_test, model.predict(X_test)),
    )

    visualizer = prediction_error(model, X_train, y_train, X_test, y_test)
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
    with open('SVR/' + str(target) + ' model.pickle', 'wb') as f:
        pickle.dump(model, f)

    return [
        model.score(X_test, y_test),
        mean_absolute_error(y_test, model.predict(X_test)),
    ]
