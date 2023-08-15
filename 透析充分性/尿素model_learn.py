from data_prepare import sufficiency_data
from model.LassoLarsCV_model import LassoLarsCV_model
from model.LightGBM_model import LightGBM_model
from model.SVR_model import SVR_model

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

# X_columns = ['性别', '原发病诊断', '治疗方案', '干体重', '透析前体重', '减衣服', '净体重', '拟脱水', '透前体温',
#        '透前收缩压', '透前舒张压', '透前心率', '透后净体重', '实际脱水', '透后体温', '透后心率', '透后收缩压',
#        '透后舒张压', '透后体重', '透后减衣服', '实际透析时间', '体重增加', '单次透析时长', '透析类型', '透析器类型',
#        '透析液流速', '透析液温度', '血液流速', '透析a液k含量', '透析a液ca含量', '透析a液na含量',
#        '透析a液hco3含量', '尿素(前)','透析时长', '年龄', '开始透析年龄' ]

Y_columns = ['尿素(后)', 'urr', 'ktv']



for target in Y_columns:
    
    if target != '尿素(后)':
        continue
    
    data = sufficiency_data[sufficiency_data[target].notnull()]
    features = data.drop(columns=Y_columns)
    Y = data[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, Y, test_size=0.2, random_state=0, shuffle=True)
    
    LassoLarsCV_model(target, X_train, y_train, X_test, y_test)
    # LightGBM_model(target, X_train, y_train, X_test, y_test)
    SVR_model(target, X_train, y_train, X_test, y_test)