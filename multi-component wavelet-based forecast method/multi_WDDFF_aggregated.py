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

# 分量预测因子
D1_iv=['Monthly_flow_t-1',
'Nino 1+2_D2_t-4',
'Monthly_rainfall_t-1',
'NP_t-6',
'AO_t-6',
'NP_D2_t-4',
'SOI_D2_t-2',
'AO_D1_t-5',
'NP_t-5',
'AO_t-5',
'NP_D1_t-5'
]
D2_iv=['Monthly_flow_D1_t-1',
'Monthly_flow_D1_t-2',
'Nino 1+2_D2_t-4',
'Monthly_flow_t-2',
'MEI_D2_t-1',
'Monthly_rainfall_t-1',
'Monthly_flow_t-1',
'SOI_D1_t-5',
'NP_t-6',
'Monthly_flow_D2_t-1',
'Monthly_rainfall_t-3',
'AO_t-6',
'Nino 1+2_D3_t-2',
'Nino 3_D2_t-4',
'MEI_D3_t-1',
'Monthly_flow_D2_t-2',
'Monthly_flow_A3_t-1',
'Monthly_flow_D3_t-2',
'AO_D2_t-4',
'NP_D2_t-6',
'Monthly_flow_A3_t-2'
]
D3_iv=['Monthly_flow_D2_t-2',
'Monthly_flow_D3_t-1',
'Monthly_flow_D2_t-1',
'Nino 1+2_D3_t-3',
'Monthly_flow_A3_t-2',
'Monthly_flow_t-1',
'Monthly_flow_A3_t-1',
'Monthly_rainfall_D3_t-1',
'NAO_A3_t-2',
'NP_D3_t-5',
'Monthly_flow_D3_t-2',
'Nino 3_D2_t-4',
'MEI_D3_t-5',
'EAWR_D1_t-3',
'Nino 1+2_D2_t-4',
'Monthly_rainfall_t-1',
'Monthly_rainfall_A3_t-1',
'Monthly_flow_t-2'
]
A3_iv=['Monthly_flow_A3_t-1',
'Monthly_flow_D3_t-1',
'Nino 1+2_D2_t-1',
'Nino 34_D2_t-2',
'Monthly_rainfall_A3_t-1',
'Monthly_flow_D3_t-2',
'Monthly_flow_A3_t-2',
'Monthly_flow_t-1',
'Nino 1+2_t-1',
'TNI_D1_t-1',
'Monthly_rainfall_D3_t-1',
'TNA_D2_t-3',
'Nino 1+2_D1_t-3',
'TNA_D3_t-2',
'Nino 3_D3_t-5',
'WHWP_A3_t-4',
'Nino 1+2_D2_t-3',
'NAO_A3_t-1',
'Nino 1+2_D2_t-2',
'Nino 3_D3_t-6'
]

# 各分量对应的预测因子个数
PMIS_iv=[D1_iv, D2_iv, D3_iv, A3_iv]
PMIS_num=[5,9,16,9]

# 评价函数
def eval_fun(sim, obs):
    return [he.rmse(sim, obs),
            he.rmsle(sim, obs),
            he.r_squared(sim, obs),
            he.nse(sim, obs),
            he.kge_2009(sim, obs),
            he.kge_2012(sim, obs)]

# 整理因子
def arange_iv():        
    data0=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\高要月径流预测\wavelet_decomposition\decomposition_result\multi_haar_3lev_predictor.xlsx',index_col=None)
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
    
    data1=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\高要月径流预测\wavelet_decomposition\decomposition_result\multi_haar_3lev_predictant.xlsx',index_col=None)
    data1=pd.DataFrame(np.array(data1.iloc[12:]),columns=data1.columns)
    potential_iv=[]
    for i in range(4):
        potential_iv.append(pd.concat([potential_iv_flow, potential_iv_rain, index_set, data1.iloc[:,i]],axis=1))
    
    return potential_iv

# add together
def elm_model(potential_iv, iter_num=1000):
    prediction_tr_=[]
    prediction_val_=[]
    prediction_te_=[]
    
    train_eval=[]
    val_eval=[]
    test_eval=[]
    
    for i in range(len(potential_iv)):
        Xtr=potential_iv[i].iloc[0:372]
        Xv=potential_iv[i].iloc[372:516]
        Xte=potential_iv[i].iloc[516:]
        
        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        scaler2.fit(np.vstack((np.array(Xtr.iloc[:,-1]).T,np.array(Xtr.iloc[:,-1]).T)).T)

        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        scaler1.fit(Xtr)
        
        Xtr=pd.DataFrame(scaler1.transform(Xtr), columns=Xtr.columns)
        Xv=pd.DataFrame(scaler1.transform(Xv), columns=Xtr.columns)
        Xte=pd.DataFrame(scaler1.transform(Xte), columns=Xtr.columns)

        Activation=['sigm']
        #base_dir=r'C:\Users\Administrator\Desktop\径流预报\实验\output\multi_WDDFF\aggregated\add together'
        #save_path=['ELM_sigm']
        
        
        for j in range(len(Activation)):
            ELM_model_set=extreme_learning_machine2(input=np.array(Xtr[PMIS_iv[i][0:PMIS_num[i]]]), observed=np.array(Xtr.iloc[:,-1]), Xv=np.array(Xv[PMIS_iv[i][0:PMIS_num[i]]]), Tv=np.array(Xv.iloc[:,-1]), dimensions_of_input=int(PMIS_num[i]), \
                        validation_strategy='V', activation_fun=Activation[j], Norm=None, num=iter_num)

            prediction_tr=0;prediction_val=0;prediction_te=0
            for k in ELM_model_set:
                prediction_tr=prediction_tr+k.predict(np.array(Xtr[PMIS_iv[i][0:PMIS_num[i]]]))
                prediction_val=prediction_val+k.predict(np.array(Xv[PMIS_iv[i][0:PMIS_num[i]]]))
                prediction_te=prediction_te+k.predict(np.array(Xte[PMIS_iv[i][0:PMIS_num[i]]]))

            prediction_tr=scaler2.inverse_transform(np.vstack(((prediction_tr/iter_num).flatten(),(prediction_tr/iter_num).flatten())).T)[:,0]
            prediction_val=scaler2.inverse_transform(np.vstack(((prediction_val/iter_num).flatten(),(prediction_val/iter_num).flatten())).T)[:,0]
            prediction_te=scaler2.inverse_transform(np.vstack(((prediction_te/iter_num).flatten(),(prediction_te/iter_num).flatten())).T)[:,0]
            
        prediction_tr_.append(prediction_tr)
        prediction_val_.append(prediction_val)
        prediction_te_.append(prediction_te)

        train_eval.append(eval_fun(prediction_tr, np.array(potential_iv[i].iloc[0:372,-1])))
        val_eval.append(eval_fun(prediction_val, np.array(potential_iv[i].iloc[372:516,-1])))
        test_eval.append(eval_fun(prediction_te, np.array(potential_iv[i].iloc[516:,-1])))
        
    final_prediction_tr=np.sum(np.array(prediction_tr_).T, axis=1)
    final_prediction_val=np.sum(np.array(prediction_val_).T, axis=1)
    final_prediction_te=np.sum(np.array(prediction_te_).T, axis=1)
    
    final_tr_eval=eval_fun(final_prediction_tr, np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[0:372]).flatten())
    final_val_eval=eval_fun(final_prediction_val, np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[372:516]).flatten())
    final_te_eval=eval_fun(final_prediction_te, np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[516:]).flatten())

    return prediction_tr_, prediction_val_, prediction_te_, final_prediction_tr, final_prediction_val, final_prediction_te,\
                    train_eval, val_eval, test_eval, final_tr_eval, final_val_eval, final_te_eval
                    
#Results0=elm_model(arange_iv(), iter_num=1000)

# ELM_ensemble
def elm_model_ens(potential_iv, iter_num=1000):
    prediction_tr_=[]
    prediction_val_=[]
    prediction_te_=[]
    
    for i in range(len(potential_iv)):
        Xtr=potential_iv[i].iloc[0:372]
        Xv=potential_iv[i].iloc[372:516]
        Xte=potential_iv[i].iloc[516:]
        
        scaler2 = MinMaxScaler(feature_range=(-1, 1))
        scaler2.fit(np.vstack((np.array(Xtr.iloc[:,-1]).T,np.array(Xtr.iloc[:,-1]).T)).T)

        scaler1 = MinMaxScaler(feature_range=(-1, 1))
        scaler1.fit(Xtr)
        
        Xtr=pd.DataFrame(scaler1.transform(Xtr), columns=Xtr.columns)
        Xv=pd.DataFrame(scaler1.transform(Xv), columns=Xtr.columns)
        Xte=pd.DataFrame(scaler1.transform(Xte), columns=Xtr.columns)

        Activation=['sigm']
        #base_dir=r'C:\Users\Administrator\Desktop\径流预报\实验\output\multi_WDDFF\aggregated\add together'
        #save_path=['ELM_sigm']
        
        
        for j in range(len(Activation)):
            ELM_model_set=HP_Elm.extreme_learning_machine2(input=np.array(Xtr[PMIS_iv[i][0:PMIS_num[i]]]), observed=np.array(Xtr.iloc[:,-1]), Xv=np.array(Xv[PMIS_iv[i][0:PMIS_num[i]]]), Tv=np.array(Xv.iloc[:,-1]), dimensions_of_input=int(PMIS_num[i]), \
                        validation_strategy='V', activation_fun=Activation[j], Norm=None, num=iter_num)

            prediction_tr=0;prediction_val=0;prediction_te=0
            for k in ELM_model_set:
                prediction_tr=prediction_tr+k.predict(np.array(Xtr[PMIS_iv[i][0:PMIS_num[i]]]))
                prediction_val=prediction_val+k.predict(np.array(Xv[PMIS_iv[i][0:PMIS_num[i]]]))
                prediction_te=prediction_te+k.predict(np.array(Xte[PMIS_iv[i][0:PMIS_num[i]]]))

            prediction_tr=(prediction_tr/iter_num).flatten()
            prediction_val=(prediction_val/iter_num).flatten()
            prediction_te=(prediction_te/iter_num).flatten()
            
        prediction_tr_.append(prediction_tr)
        prediction_val_.append(prediction_val)
        prediction_te_.append(prediction_te)
    
    
    
    
    obs_tr=np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[0:372]).flatten()
    obs_val=np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[372:516]).flatten()
    obs_te=np.array(IVS_PMI_output_csv_for_input.Flow_data_monthly2.iloc[516:]).flatten()
    
    scaler3 = MinMaxScaler(feature_range=(-1, 1))
    scaler3.fit(np.vstack((np.array(obs_tr).T,np.array(obs_tr).T)).T)
    
    ELM_model_set2=HP_Elm.extreme_learning_machine2(input=np.array(prediction_tr_).T, observed=scaler3.transform(np.vstack((obs_tr,obs_tr)).T)[:,0], Xv=np.array(prediction_val_).T, Tv=scaler3.transform(np.vstack((obs_val,obs_val)).T)[:,0], dimensions_of_input=int(len(prediction_tr_)), \
                    validation_strategy='V', activation_fun=Activation[0], Norm=None, num=iter_num)
    prediction_tr2=0;prediction_val2=0;prediction_te2=0
    for k in ELM_model_set2:
        prediction_tr2=prediction_tr2+k.predict(np.array(prediction_tr_).T)
        prediction_val2=prediction_val2+k.predict(np.array(prediction_val_).T)
        prediction_te2=prediction_te2+k.predict(np.array(prediction_te_).T)

    prediction_tr2=(prediction_tr2/iter_num).flatten()
    prediction_val2=(prediction_val2/iter_num).flatten()
    prediction_te2=(prediction_te2/iter_num).flatten()
    
    prediction_tr2=scaler3.inverse_transform(np.vstack((prediction_tr2,prediction_tr2)).T)[:,0]
    prediction_val2=scaler3.inverse_transform(np.vstack((prediction_val2,prediction_val2)).T)[:,0]
    prediction_te2=scaler3.inverse_transform(np.vstack((prediction_te2,prediction_te2)).T)[:,0]

    final_tr_eval=eval_fun(prediction_tr2, obs_tr)
    final_val_eval=eval_fun(prediction_val2, obs_val)
    final_te_eval=eval_fun(prediction_te2, obs_te)

    return prediction_tr2,prediction_val2,prediction_te2,\
                    final_tr_eval, final_val_eval, final_te_eval

#Results=elm_model_ens(arange_iv(), iter_num=1000)
















