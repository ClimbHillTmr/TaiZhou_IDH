import pandas as pd
import numpy as np

import sys

sys.path.append("..")

from collections import Counter

# 数据读取
## 透析记录
DIALYSIS = pd.read_excel('../Dataset/台州中心透析记录.xlsx')
## 透析处方
DOCTORS_ADVICE = pd.read_excel('../Dataset/台州中心医院-HD-透析处方.xls')
## 检测结果
LIS_REPORT = pd.read_excel('../Dataset/透析数据--台州市中心医院.xls', sheet_name=None)

# 日期格式转换
DIALYSIS['透析日期'] = pd.to_datetime(DIALYSIS['透析日期'])
DIALYSIS['出生日期'] = pd.to_datetime(DIALYSIS['出生日期'])

DOCTORS_ADVICE['透析日期'] = pd.to_datetime(DOCTORS_ADVICE['透析日期'])
DOCTORS_ADVICE['出生日期'] = pd.to_datetime(DOCTORS_ADVICE['出生日期'])

## 透析记录合并透析处方
DIALYSIS_RECORD = []
# DIALYSIS = DIALYSIS.sort_values(by='透析日期')

for i in range(len(DIALYSIS)):
    if i % 1000 == 0:
        print('加载数据', str(round(i / len(DIALYSIS) * 100, 2)) + '%')
    patient_date = DIALYSIS['透析日期'][i]
    patient_id = DIALYSIS['病人姓名'][i]
    patient_type = DIALYSIS['治疗方案'][i]

    record = list(DIALYSIS.loc[i])

    advice_group = DOCTORS_ADVICE[(DOCTORS_ADVICE['病人姓名'] == patient_id)]
    if len(advice_group) == 0:
        continue
    for idx,advice in advice_group.iterrows():
        if advice['透析日期'] <= patient_date and advice['透析类型'] == patient_type:
            prescription = advice
        # 匹配当天的处方
        if prescription['透析日期'] == patient_date and prescription['透析类型'] == patient_type:
            break
        if advice['透析日期'] > prescription['透析日期'] and prescription['透析类型'] == patient_type:
            break
    records = record + list(prescription)
    DIALYSIS_RECORD.append(records)

DIALYSIS_RECORD = pd.DataFrame(DIALYSIS_RECORD)

DIALYSIS_RECORD.to_csv('DIALYSIS_RECORD.csv')

DIALYSIS_RECORD.columns = list(DIALYSIS.columns) + list(DOCTORS_ADVICE.columns)
DIALYSIS_RECORD = DIALYSIS_RECORD.loc[:, ~DIALYSIS_RECORD.columns.duplicated()]

# common_columns = set(DIALYSIS.columns.tolist()) & set(DOCTORS_ADVICE.columns.tolist())
# filtered_columns = common_columns - set(['身份证', '透析日期'])
# DIALYSIS_RECORD  = pd.merge(DIALYSIS, DOCTORS_ADVICE.drop(columns=filtered_columns), on=['身份证', '透析日期'])

# 一段时间调整

# # 维持性透析患者
# from utils import filter_by_date_range

# df_grouped = DIALYSIS.groupby('病人姓名')
# Maintenance_Duration = df_grouped.apply(filter_by_date_range)
# Maintenance_Dialysis =
