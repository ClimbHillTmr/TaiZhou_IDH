import pandas as pd

import datetime

def filter_by_date_range(groups):
    group = groups['透析日期']
    # 计算最旧日期和最新日期的差值
    date_range = datetime.datetime(group.max()) - datetime.datetime(group.min())
    # 如果差值大于90天，则返回该组数据
    if date_range > pd.Timedelta(90, 'd'):
        return group

