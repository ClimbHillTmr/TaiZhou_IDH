import pandas as pd
import numpy as np
from sklearn import preprocessing

import sys

sys.path.append("..")

from DataPreprocessing.data_load import DIALYSIS_RECORD
from DataPreprocessing.utils import drop_null

# Load data
LIS_REPORT = pd.read_excel('../Dataset/透析数据--台州市中心医院.xls', sheet_name=None)
kidney = LIS_REPORT['肾功能'].rename(columns={'检验日期': '透析日期'})
sufficiency = LIS_REPORT['透析充分性']

# Preprocess dates
date_columns = ['透析日期', '出生日期', '首次透析日期']
for col in date_columns:
    sufficiency[col] = pd.to_datetime(sufficiency[col]).dt.normalize()
    kidney[col] = pd.to_datetime(sufficiency[col]).dt.normalize()

# Merge data
target_columns = [
    '透析日期',
    '身份证',
    '肌酐(前)',
    '肌酐(后)',
    '尿素(前)',
    '尿素(后)',
    '尿酸(前)',
    '尿酸(后)',
    'urr',
]
results = pd.merge(kidney[target_columns], sufficiency, on=['透析日期', '身份证'], how='outer')

# Fill missing values
fill_columns = ['尿素(前)', '尿素(后)', 'urr']
for col in fill_columns:
    results[col] = results[f'{col}_x'].fillna(results[f'{col}_y'])

results.drop(
    [f'{col}_x' for col in fill_columns] + [f'{col}_y' for col in fill_columns],
    axis=1,
    inplace=True,
)

# Merge with DIALYSIS_RECORD
common_columns = set(DIALYSIS_RECORD.columns) & set(results.columns)
filtered_columns = common_columns - {'身份证', '透析日期'}

dataset = pd.merge(
    DIALYSIS_RECORD, results.drop(columns=filtered_columns), on=['身份证', '透析日期']
)

# Calculate derived features
date_diff_columns = ['透析时长_天', '年龄_天', '开始透析年龄_天', '透析时长_年', '年龄_年', '开始透析年龄_年']
date_columns = ['透析日期', '首次透析日期', '出生日期']
for diff_col, date_col in zip(date_diff_columns, date_columns):
    dataset[diff_col] = (dataset[date_col] - dataset['出生日期']).dt.days
date_diff_columns = ['透析时长_年', '年龄_年', '开始透析年龄_年']
date_columns = ['透析日期', '首次透析日期', '出生日期']
for diff_col, date_col in zip(date_diff_columns, date_columns):
    dataset[diff_col] = (
        pd.to_datetime(dataset[date_col]).dt.year
        - pd.to_datetime(dataset['出生日期']).dt.year
    )

dataset['透析前体重-干体重'] = dataset['透析前体重'] - dataset['干体重']
dataset['拟脱水差'] = dataset['拟脱水'] - dataset['透析前体重-干体重']


# Drop unnecessary columns
columns_to_drop = ['病人姓名', '透析号', '病例号', '医生总结', '护士总结', '首次透析日期', "透析日期", "出生日期"]
dataset.drop(columns=columns_to_drop, inplace=True)

# Apply Label Encoding
one_hot_columns = ['性别', '原发病诊断', '治疗方案', '是否诱导方案', '透析器类型', '透析类型']
encoder = preprocessing.LabelEncoder()
for col in one_hot_columns:
    dataset[col].fillna(dataset[col].mode()[0], inplace=True)  # Corrected this line
    dataset[col] = encoder.fit_transform(dataset[col])

dataset['id'] = encoder.fit_transform(dataset['身份证'])
dataset.drop(columns=['身份证'], inplace=True)

# Handle missing values (removed the loop for this step)

# Remove negative and zero values
non_negative_columns = ['干体重', '拟脱水']
dataset = dataset[(dataset[non_negative_columns] > 0).all(axis=1)]

# Calculate percentiles for outlier handling
percentiles = [0.01, 0.99]
percentile_values = dataset.quantile(percentiles)
dataset = dataset.apply(
    lambda x: np.clip(
        x,
        percentile_values.loc[percentiles[0], x.name],
        percentile_values.loc[percentiles[1], x.name],
    )
)

# Prepare target columns
target_columns = ['尿素(后)', 'urr', 'ktv', '肌酐(后)', '尿素(后)']
target_data = dataset[target_columns]
dataset.drop(columns=target_columns, inplace=True)

# Concatenate target columns
final_dataset = pd.concat([dataset, target_data], axis=1)
print(final_dataset.info())