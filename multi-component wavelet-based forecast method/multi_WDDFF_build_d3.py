# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
import HydroErr as he
from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import read_data
import HP_Elm

# 整理因子
def arange_iv():        
    data0=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\wavelet_decomposition_with_Matlab\decomposition_result\multi_haar_3lev_predictor(hybrid_across_with_D1-D3、A3).xlsx',index_col=None)
    data0=data0.iloc[6:]
    
    potential_iv_flow= pd.concat([pd.DataFrame(np.array(data0.iloc[5:-1,0:5]),columns=np.array(data0.iloc[:,0:5].columns)+np.array(5*['_t-1'])), \
                    pd.DataFrame(np.array(data0.iloc[4:-2,0:5]),columns=np.array(data0.iloc[:,0:5].columns)+np.array(5*['_t-2']))], axis=1)

    potential_iv_rain= pd.concat([pd.DataFrame(np.array(data0.iloc[5:-1,5:10]),columns=np.array(data0.iloc[:,5:10].columns)+np.array(5*['_t-1'])), \
                    pd.DataFrame(np.array(data0.iloc[4:-2,5:10]),columns=np.array(data0.iloc[:,5:10].columns)+np.array(5*['_t-2'])),\
                        pd.DataFrame(np.array(data0.iloc[3:-3,5:10]),columns=np.array(data0.iloc[:,5:10].columns)+np.array(5*['_t-3']))], axis=1)
    
    index_set=[]
    for i in range(6):
        index_set.append(pd.DataFrame(np.array(data0.iloc[5-i:-1-i,10:]),columns=np.array(data0.iloc[:,10:].columns)+np.array(100*['_t-'+str(i+1)])))
    index_set=pd.concat(index_set,axis=1)
    
    data1=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\wavelet_decomposition_with_Matlab\decomposition_result\multi_haar_3lev_predictant(hybrid_across_with_D1-D3、A3).xlsx',index_col=None)
    data1=pd.DataFrame(np.array(data1.iloc[12:]),columns=data1.columns)
    potential_iv=[]
    for i in range(4):
        potential_iv.append(pd.concat([potential_iv_flow, potential_iv_rain, index_set, data1.iloc[:,i]],axis=1))
    
    return potential_iv
    
# 数据集划分
'''
    训练集：1958-01~1988-12 (372 samples, 60.8%)
    验证集：1989-01~2000-12 (144 samples, 23.5%)
    测试集：2001-01~2008-12 (96 samples, 15.7%)
'''

potential_iv=arange_iv()[2]
potential_iv_tr=potential_iv.iloc[0:372]
potential_iv_val=potential_iv.iloc[372:516]
potential_iv_te=potential_iv.iloc[516:]

# d3分量的因子(top20)
PMIS_iv=['Monthly_flow_D2_t-2',
'Monthly_flow_D3_t-1',
'Monthly_flow_D2_t-1',
'Nino 1+2_D3_t-2',
'Monthly_flow_D1_t-1',
'Monthly_flow_A3_t-1',
'Monthly_flow_t-1',
'Monthly_flow_D3_t-2',
'Monthly_rainfall_D2_t-3',
'Monthly_flow_t-2',
'Nino 1+2_t-1',
'Monthly_flow_A3_t-2',
'SOI_D2_t-3',
'SOI_D1_t-6',
'MEI_D1_t-4',
'ONI_D3_t-1',
'Monthly_rainfall_D3_t-1',
'WHWP_D3_t-5',
'Nino 4_D1_t-3',
'AO_D1_t-5'
]

# 评价函数
def eval_fun(sim, obs):
    return [he.rmse(sim, obs),
            he.rmsle(sim, obs),
            he.r_squared(sim, obs),
            he.nse(sim, obs),
            he.kge_2009(sim, obs),
            he.kge_2012(sim, obs)]

def elm_model(Xtr, Xv, Xte, iter_num=1000):

    scaler2 = MinMaxScaler(feature_range=(-1, 1))
    scaler2.fit(np.vstack((np.array(Xtr.iloc[:,-1]).T,np.array(Xtr.iloc[:,-1]).T)).T)

    scaler1 = MinMaxScaler(feature_range=(-1, 1))
    scaler1.fit(Xtr)
    Xtr=pd.DataFrame(scaler1.transform(Xtr), columns=potential_iv.columns)
    Xv=pd.DataFrame(scaler1.transform(Xv), columns=potential_iv.columns)
    Xte=pd.DataFrame(scaler1.transform(Xte), columns=potential_iv.columns)

    Activation=['lin', 'sigm', 'tanh', 'rbf_l1', 'rbf_l2', 'rbf_linf']         # 使用不同核函数进行拟合
    input_num=np.linspace(2,20,19)

    train_eval=[]
    val_eval=[]
    test_eval=[]
    index=[]

    for i in range(len(Activation)):
        for j in input_num:
            j=int(j)
            ELM_model_set=HP_Elm.extreme_learning_machine2(input=np.array(Xtr[PMIS_iv[:j]]), observed=np.array(Xtr.iloc[:,-1]), Xv=np.array(Xv[PMIS_iv[:j]]), Tv=np.array(Xv.iloc[:,-1]), dimensions_of_input=j, \
                        validation_strategy='V', activation_fun=Activation[i], Norm=None, num=iter_num)

            prediction_tr=0;prediction_val=0;prediction_te=0
            for k in ELM_model_set:
                prediction_tr=prediction_tr+k.predict(np.array(Xtr[PMIS_iv[:j]]))
                prediction_val=prediction_val+k.predict(np.array(Xv[PMIS_iv[:j]]))
                prediction_te=prediction_te+k.predict(np.array(Xte[PMIS_iv[:j]]))

            prediction_tr=prediction_tr/iter_num
            prediction_val=prediction_val/iter_num
            prediction_te=prediction_te/iter_num

            train_eval.append(eval_fun(scaler2.inverse_transform(np.hstack((prediction_tr,prediction_tr)))[:,0], np.array(potential_iv_tr.iloc[:,-1])))
            val_eval.append(eval_fun(scaler2.inverse_transform(np.hstack((prediction_val,prediction_val)))[:,0], potential_iv_val.iloc[:,-1]))
            test_eval.append(eval_fun(scaler2.inverse_transform(np.hstack((prediction_te,prediction_te)))[:,0], potential_iv_te.iloc[:,-1]))

            index.append(Activation[i]+'_input'+str(j))
        print(str(Activation[i])+' is completed!')

    return pd.DataFrame(np.array(train_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012'], index=index),\
        pd.DataFrame(np.array(val_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012'], index=index),\
            pd.DataFrame(np.array(test_eval), columns=['rmse','rmsle','r_squared','nse','kge_2009','kge_2012'], index=index)
            
if __name__ == "__main__":
    tr_eval,val_eval,te_eval=elm_model(potential_iv_tr, potential_iv_val, potential_iv_te, iter_num=200)


