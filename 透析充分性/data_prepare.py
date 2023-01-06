import pandas as pd 
import numpy as np

import sys 
sys.path.append("..") 

from DataPreprocessing.data_load import DIALYSIS_RECORD, LIS_REPORT

sufficiency = LIS_REPORT['透析充分性']

sufficiency['透析日期'] = pd.to_datetime(sufficiency['透析日期']).dt.normalize()
sufficiency['出生日期'] = pd.to_datetime(sufficiency['出生日期']).dt.normalize()
sufficiency['首次透析日期'] = pd.to_datetime(sufficiency['首次透析日期']).dt.normalize()


common_columns = set(DIALYSIS_RECORD.columns.tolist()) & set(LIS_REPORT['透析充分性'].columns.tolist())

# filtered_columns = common_columns - set(['病人姓名', '透析日期'])

# dataset  = pd.merge(DIALYSIS_RECORD, LIS_REPORT['透析充分性'].drop(columns=filtered_columns), on=['病人姓名', '透析日期'])

filtered_columns = common_columns - set(['病人姓名', '透析日期'])

dataset  = pd.merge(DIALYSIS_RECORD, sufficiency.drop(columns=filtered_columns), on=['病人姓名', '透析日期'])

print(dataset['尿素(后)'].describe())

dataset.columns

dataset.to_csv('dataset.csv')

sufficiency_dataset = dataset

sufficiency_dataset['透析时长'] = sufficiency_dataset.apply(lambda row: (row["透析日期"] - row["首次透析日期"]).days, axis=1)
sufficiency_dataset['年龄'] = sufficiency_dataset.apply(lambda row: (row["透析日期"] - row["出生日期"]).days, axis=1)
sufficiency_dataset['开始透析年龄'] = sufficiency_dataset.apply(lambda row: (row["首次透析日期"] - row["出生日期"]).days, axis=1)

columns_to_drop = ['病人姓名','透析号','病例号','医生总结', '护士总结', '是否诱导方案', '方案频率','首次透析日期',"透析日期","出生日期"]

sufficiency_dataset = dataset.drop(columns=columns_to_drop)

# OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
one_hot_columns = ['性别','原发病诊断','治疗方案', '透析器类型']

for i in one_hot_columns:
    enc.fit(sufficiency_dataset[i])