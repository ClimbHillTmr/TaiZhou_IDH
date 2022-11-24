import pandas as pd 
import numpy as np

import sys 
sys.path.append("..") 

DIALYSIS = pd.read_excel('../Dataset/台州中心透析记录.xls')

DOCTORS_ADVICE = pd.read_excel('../Dataset/台州中心医院-HD-透析处方.xls')

LIS_REPORT = pd.read_excel('../Dataset/透析数据--台州市中心医院.xls')