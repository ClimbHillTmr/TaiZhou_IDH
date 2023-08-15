import pandas as pd

from data_prepare import final_dataset
from model.LassoLarsCV_model import LassoLarsCV_model
from model.LightGBM_model import LightGBM_model
from model.XGBoost_model import XGB_model
from model.SVR_model import SVR_model
from model.FeatureSelectwithML import FeatureSelectwithML

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import smogn

features_columns = [
    '性别',
    '原发病诊断',
    '治疗方案',
    '干体重',
    '透析前体重',
    '减衣服',
    '净体重',
    '拟脱水',
    '透前体温',
    '透析前体重-干体重',
    '透前收缩压',
    '透前舒张压',
    '透前心率',
    '透析液流速',
    '透析液温度',
    '血液流速',
    '方案频率',
    '透析a液k含量',
    '透析a液ca含量',
    '透析a液na含量',
    '透析a液hco3含量',
    # '尿素(前)',
    '透析时长_天',
    '年龄_天',
    '开始透析年龄_天',
    '透析时长_年',
    '年龄_年',
    '开始透析年龄_年',
    '单次透析时长',
    '透析类型',
    '透析器类型',
    # 'id',
    # '尿酸(前)',
    # '尿酸(后)',
    # 'urr',
    # 'ktv',
    # '肌酐(后)',
    # '尿素(后)',
]

target = [
    #   '尿酸(后)',
    'urr',
    'ktv',
    # '肌酐(后)',
    # '尿素(后)',
]

Y_columns = 'urr'

data = final_dataset[features_columns + target].dropna()

ranks, features_rank_Select = FeatureSelectwithML(
    features_columns, data.drop(target, axis=1), data[Y_columns], classify=False
)

results = []  # List to store model results

for i in range(len(ranks.index)):
    X_columns = list(ranks.index)[: i + 1]
    X = data[X_columns]

    # Iterate through columns
    for col in X.columns:
        # Check if the column has missing values
        if X[col].isnull().any():
            # Check if the column data type is integer
            if X[col].dtype == 'int64':
                X[col].fillna(X[col].mode().iloc[0], inplace=True)
            # Check if the column data type is float
            elif X[col].dtype == 'float64':
                X[col].fillna(X[col].mean(), inplace=True)

    stdsc = StandardScaler().fit(X)  # 正态分布标准化数据的函数
    X = stdsc.fit_transform(X)  # 都正态分布标准化

    X_train, X_test, y_train, y_test = train_test_split(
        X, data[Y_columns], test_size=0.2, random_state=0, shuffle=True
    )

    Train_dataset = pd.concat(
        [
            pd.DataFrame(X_train).reset_index(drop=True),
            pd.DataFrame(y_train).reset_index(drop=True),
        ],
        axis=1,
    )

    Train_smogn = smogn.smoter(
        data=Train_dataset,
        k=5,
        samp_method='extreme',
        y=Y_columns,
        # rel_method='auto',
    )

    X_train, y_train = Train_dataset.drop(Y_columns, axis=1), Train_dataset[Y_columns]

    ll_result = LassoLarsCV_model(target, X_train, y_train, X_test, y_test)
    svr_result = SVR_model(target, X_train, y_train, X_test, y_test)
    lgbm_result = LightGBM_model(target, X_train, y_train, X_test, y_test)
    # xgb_result = XGB_model(target, X_train, y_train, X_test, y_test)

    # Store the results in the results list
    results.append(
        {
            'num_features': i + 1,
            'LassoLarsCV': ll_result,
            'SVR': svr_result,
            'LightGBM': lgbm_result,
            # 'XGBoost': xgb_result,
        }
    )
# Analyze the results to find the best model and feature count
