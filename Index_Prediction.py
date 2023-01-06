import pandas as pd
import numpy as np

import datetime

from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

class HD_index_prediction:
    def __init__(self):
        pass
    def load_data(self,info_data,X_data,Y_data):
        info_data[['HOSPITAL_ID', 'PATIENT_NK']] = info_data[['HOSPITAL_ID', 'PATIENT_NK']].astype(str)
        #info_data=info_data[info_data['AGE']>18]
        
        X_data[['HOSPITAL_ID', 'PATIENT_NK']] = X_data[['HOSPITAL_ID', 'PATIENT_NK']].astype(str)


        self.X_data=X_data
        self.Y_data=Y_data
        self.info_data=info_data
    def preprocess_training(self,index):
        self.index=index
        time_limit=60
        delta_time=datetime.timedelta(days=time_limit)
        X_list=[]
        Y_list=[]
        patient_list=[]
        index_average_value_list=[]
        for i in range(len(self.Y_data)):
            if i%1000==0:
                print('加载数据',str(round(i/len(self.Y_data)*100,2))+'%')

            if eval(Y_data[index].iloc[i])==[]:
                continue
            
            p=self.Y_data['PATIENT_NK'].iloc[i]
            h=self.Y_data['HOSPITAL_ID'].iloc[i]
            each_X=self.X_data[(self.X_data['PATIENT_NK']==p)&(self.X_data['HOSPITAL_ID']==h)]
            print(each_X)
            if len(each_X)==0:
                continue
            
            each_info=self.info_data[(self.info_data['PATIENT_NK']==p)&(self.info_data['HOSPITAL_ID']==h)]
            if len(each_info)==0:
                continue
            
            age=each_info['AGE'].iloc[0]
            gender=each_info['GENDER'].iloc[0]
            
            if gender==1:
                gender=1
            else:
                gender=0
            HD_time_list=list(each_X['CREATE_TIME'])
            time_list=eval(self.Y_data[index+'日期'].iloc[i])
            if len(time_list)<5:
                continue
            
            time_matrix=np.zeros((len(time_list)-4,len(HD_time_list)))
            
            for l in range(4,len(time_list)):
                for j in range(len(HD_time_list)):
                    HD_time=datetime.datetime.strptime(HD_time_list[j], "%Y-%m-%d %H.%M.%S")
                    index_time=datetime.datetime.strptime(time_list[l], "%Y-%m-%d %H.%M.%S")
                    value=max((index_time-HD_time).days,0)
                    if value==0:
                        value=999999#这样就取不到这个时间点
                    time_matrix[l-4,j]=value
            
            HD_index_list=[]
            j_list=[]
            for l in range(len(time_list)-4):
                if np.min(time_matrix[l])<time_limit:
                    j=np.where(time_matrix[l]==np.min(time_matrix[l]))[0][0]
                    if j in j_list:
                        continue
                    j_list.append(j)
                    HD_index_list.append([l+4,j])
            data_list=eval(self.Y_data[index].iloc[i])
        
        
            for lj in HD_index_list:
                l=lj[0]
                j=lj[1] 
                HD_time=each_X['CREATE_TIME'].iloc[j]
                HD_time=datetime.datetime.strptime(HD_time, "%Y-%m-%d %H.%M.%S")
                
                index_time=time_list[l]
                index_time=datetime.datetime.strptime(index_time, "%Y-%m-%d %H.%M.%S")
                if index_time>HD_time and index_time-HD_time<delta_time:
                    each_x=[]
                    each_x=each_x
                    if each_X['NEOPATHY_TYPE'].iloc[j]!=0:
                        LBP=1
                    else:
                        LBP=0
                    DBP_list=eval(each_X['DBP'].iloc[j].replace('nan','0'))
                    MEAN_AP_list=eval(each_X['MEAN_AP'].iloc[j].replace('nan','0'))
                    last_SBP_list=eval(each_X['SBP'].iloc[j].replace('nan','0'))
        
                    last_SBP_list=np.array(last_SBP_list)
                    last_SBP_list=last_SBP_list[last_SBP_list>0]
                    
                    DBP_list=np.array(DBP_list)
                    DBP_list=DBP_list[DBP_list>0]
                    if len(last_SBP_list)<2 or len(DBP_list)<2:
                        continue
                    
                    SBP_mean=np.mean(last_SBP_list)
                    SBP_std=np.std(last_SBP_list)
                    SBP_diff=last_SBP_list[:-1]-last_SBP_list[1:]
                    SBP_diff_abs_mean = np.mean(np.abs(SBP_diff))
                    SBP_diff_abs_std= np.std(np.abs(SBP_diff))
                    SBP_diff_mean = np.mean(SBP_diff)
                    SBP_diff_std = np.std(SBP_diff)
                    
                    DBP_mean=np.mean(DBP_list)
                    DBP_std=np.std(DBP_list)
                    DBP_diff=DBP_list[:-1]-DBP_list[1:]
                    DBP_diff_abs_mean = np.mean(np.abs(DBP_diff))
                    DBP_diff_abs_std= np.std(np.abs(DBP_diff))
                    DBP_diff_mean = np.mean(DBP_diff)
                    DBP_diff_std = np.std(DBP_diff)
                    
                    MEAN_AP_list=np.array(MEAN_AP_list)
                    MEAN_AP_list=MEAN_AP_list[MEAN_AP_list>0]
                    
                    MEAN_AP_mean=np.mean(MEAN_AP_list)
                    MEAN_AP_std=np.std(MEAN_AP_list)
        
                    each_x=each_x+[SBP_mean,SBP_std,
                              SBP_diff_abs_mean,SBP_diff_abs_std,
                              SBP_diff_mean,SBP_diff_std,
                              DBP_mean,DBP_std,
                              DBP_diff_abs_mean,DBP_diff_abs_std,
                              DBP_diff_mean,DBP_diff_std,
                              MEAN_AP_mean,MEAN_AP_std,
                              age,data_list[l-4],data_list[l-3],data_list[l-2],data_list[l-1]]
             
                    each_x=each_x + [gender,LBP]
                    if data_list[l]<=0:
                        continue
                    X_list.append(each_x)
                    
                    Y_list.append(data_list[l])
                    if [h,p] not in patient_list:
                        index_average_value_list.append(np.mean(data_list))
                        patient_list.append([h,p])
                        
        
        X_list=np.array(X_list)
        Y_list=np.array(Y_list)
        stdsc=StandardScaler().fit(X_list[:,:-2])
        np.savetxt('数据基础：特征值'+str(index)+'.csv', X_list, delimiter=",")
        np.savetxt('数据基础：目标值'+str(index)+'.csv', Y_list, delimiter=",")
        X_std=stdsc.fit_transform(X_list[:,:-2])#正态分布标准化
        X_std = np.concatenate((X_std,X_list[:,-2:]),axis=1)
        
        
        X_train,X_test,y_train,y_test= train_test_split(X_std,Y_list,test_size=0.2,random_state=0,shuffle=True)
        #X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=0.25,random_state=0,shuffle=True)
        
        lr = linear_model.LassoLarsCV(cv=2,n_jobs=-1)
        lr.fit(X_train, y_train)
        lr.score(X_train,y_train)
        
        # score = r2_score(y_train, lr.predict(X_train)),
                 

        self.model=lr
        self.stdsc=stdsc
        self.score=score
        
        return X_list,Y_list,score
    
    def predict(self,x):
        X_std=self.stdsc.fit_transform(x[:,:-2])#正态分布标准化
        
        X_std = np.concatenate((X_std,x[:,-2:]),axis=1)
        y_pred = self.model.predict(X_std)
        return y_pred