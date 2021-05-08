# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
import HydroErr as he
from sklearn.preprocessing import MinMaxScaler
from sklearn import linear_model
import pickle

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 添加文件所在文件夹
import IVS_PMI_output_csv_for_DDFF_input


# 数据集划分
'''
    训练集：1958-01~1988-12 (372 samples, 60.8%)
    验证集：1989-01~2000-12 (144 samples, 23.5%)
    测试集：2001-01~2008-12 (96 samples, 15.7%)
'''
potential_iv=IVS_PMI_output_csv_for_DDFF_input.arrange_potential_iv()
potential_iv=pd.concat([potential_iv, pd.DataFrame(IVS_PMI_output_csv_for_DDFF_input.Flow_data_monthly2.values.flatten(),columns=['target'])],axis=1)
potential_iv_tr=potential_iv.iloc[0:372]
potential_iv_val=potential_iv.iloc[372:516]
potential_iv_te=potential_iv.iloc[516:]

# PMIS挑选得到的top20
PMIS_iv=['flow_t-1',
'Nino 1+2_t-3',
'NP_t-6',
'AO_t-5',
'AO_t-6',
'rain_t-1',
'ONI_t-6',
'AO_t-4',
'SOI_t-5',
'NP_t-5',
'MEI_t-2',
'Nino 1+2_t-4',
'ONI_t-5',
'Nino 3_t-3',
'SOI_t-2',
'MEI_t-3',
'SOI_t-4',
'NP_t-4',
'MEI_t-1',
'Nino 3_t-4'
]

# 评价函数
def eval_fun(sim, obs):
    return [he.rmse(sim, obs),
            he.rmsle(sim, obs),
            he.r_squared(sim, obs),
            he.nse(sim, obs),
            he.kge_2009(sim, obs),
            he.kge_2012(sim, obs),
            he.mae(sim, obs)]
 
 
 
# ************************************************************************************************
# 构建MLR模型
def MLR(Xtr, Xv, Xte, data_save_path=None):
    
    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    scaler2.fit(np.vstack((np.array(Xtr.iloc[:,-1]).T,np.array(Xtr.iloc[:,-1]).T)).T)
    
    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler1.fit(Xtr)
    Xtr=pd.DataFrame(scaler1.transform(Xtr), columns=potential_iv.columns)
    Xv=pd.DataFrame(scaler1.transform(Xv), columns=potential_iv.columns)
    Xte=pd.DataFrame(scaler1.transform(Xte), columns=potential_iv.columns)
    
    input_num=np.linspace(2,20,19)

    train_eval=[]
    val_eval=[]
    test_eval=[]
    index=[]
    
    prediction_tr_=[]
    prediction_val_=[]
    prediction_te_=[]
    
    for j in input_num:
        j=int(j)
        reg = linear_model.LinearRegression(normalize=False)
        
        reg_=reg.fit(np.array(Xtr[PMIS_iv[:j]]), np.array(Xtr['target']))
        
        prediction_tr=reg_.predict(np.array(Xtr[PMIS_iv[:j]]))
        prediction_val=reg_.predict(np.array(Xv[PMIS_iv[:j]]))
        prediction_te=reg_.predict(np.array(Xte[PMIS_iv[:j]]))
        
        train_eval.append(eval_fun(scaler2.inverse_transform(np.vstack((prediction_tr,prediction_tr)).T)[:,0], np.array(potential_iv_tr.iloc[:,-1])))
        val_eval.append(eval_fun(scaler2.inverse_transform(np.vstack((prediction_val,prediction_val)).T)[:,0], potential_iv_val.iloc[:,-1]))
        test_eval.append(eval_fun(scaler2.inverse_transform(np.vstack((prediction_te,prediction_te)).T)[:,0], potential_iv_te.iloc[:,-1]))
        
        prediction_tr_.append(scaler2.inverse_transform(np.vstack((prediction_tr,prediction_tr)).T)[:,0])
        prediction_val_.append(scaler2.inverse_transform(np.vstack((prediction_val,prediction_val)).T)[:,0])
        prediction_te_.append(scaler2.inverse_transform(np.vstack((prediction_te,prediction_te)).T)[:,0])
        
        index.append('input_num'+str(j))
    
    train_eval=pd.DataFrame(np.array(train_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    val_eval=pd.DataFrame(np.array(val_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    test_eval=pd.DataFrame(np.array(test_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    
    # 保存计算结果
    fw=open(data_save_path, 'wb')
    pickle.dump(prediction_tr_, fw); pickle.dump(prediction_val_, fw); pickle.dump(prediction_te_, fw)
    pickle.dump(train_eval, fw); pickle.dump(val_eval, fw); pickle.dump(test_eval, fw)
    fw.close()
            
    return train_eval, val_eval, test_eval
 
if __name__ == "__main__":
    tr_eval,val_eval,te_eval=MLR(potential_iv_tr, potential_iv_val, potential_iv_te, data_save_path=r'I:\预报与不确定性资料\Monthly streamflow prediction using a stochastic data-driven framework\实验\output\博罗月径流预测2\DDFF\ddff_mlr_results.txt')
