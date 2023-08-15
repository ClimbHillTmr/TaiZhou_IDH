import pandas as pd

import datetime


def filter_by_date_range(groups):
    group = groups['透析日期']
    # 计算最旧日期和最新日期的差值
    date_range = datetime.datetime(group.max()) - datetime.datetime(group.min())
    # 如果差值大于90天，则返回该组数据
    if date_range > pd.Timedelta(90, 'd'):
        return group


def drop_null(data, percent):

    null_val_sums = data.isnull().sum()  # 统计每个列有多少缺失值

    per_null = list(null_val_sums.values / len(data))  # 计算缺失率

    no_null_name = []

    feature_name = list(null_val_sums.index)

    for i in range(len(per_null)):

        if per_null[i] < percent:

            no_null_name.append(i)

    newList = []

    for index in no_null_name:

        newList.append(feature_name[index])

    for aVal in newList:

        feature_name.remove(aVal)

    data_drop = data.drop(feature_name, axis=1)

    return data_drop
