# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
import HydroErr as he
from sklearn.preprocessing import MinMaxScaler
import pickle

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 添加文件所在文件夹
import IVS_PMI_output_csv_for_DDFF_input

base_dir=r'C:\Users\Administrator\Desktop\径流预报\实验\output'

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
# 构建ELM模型
def get_performance_index():
    
    # 获取结果
    fr=open(base_dir+'\博罗月径流预测2\DDFF'+'\ddff_results.txt', 'rb')
    ens=[]
    for j in range(3):
        ens.append(pickle.load(fr))
    fr.close()
    
    Activation=['lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']         # 使用不同核函数进行拟合
    input_num=np.linspace(2,20,19)

    train_eval=[]
    val_eval=[]
    test_eval=[]
    index=[]
    
    prediction_tr_=[]
    prediction_val_=[]
    prediction_te_=[]
    
    for i in range(len(Activation)):
        for j in range(len(input_num)):
            j=int(j)
            train_eval.append(eval_fun(ens[0][i*19+j], np.array(potential_iv_tr.iloc[:,-1])))
            val_eval.append(eval_fun(ens[1][i*19+j], potential_iv_val.iloc[:,-1]))
            test_eval.append(eval_fun(ens[2][i*19+j], potential_iv_te.iloc[:,-1]))
            index.append(Activation[i]+'_input'+str(j+2))
    
    train_eval=pd.DataFrame(np.array(train_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    val_eval=pd.DataFrame(np.array(val_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    test_eval=pd.DataFrame(np.array(test_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012','mae'], index=index)
    
            
    return train_eval, val_eval, test_eval
 
if __name__ == "__main__":
    tr_eval,val_eval,te_eval=get_performance_index()
 
 

