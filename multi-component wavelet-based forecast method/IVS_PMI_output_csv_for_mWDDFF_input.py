# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import HydroErr as he

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import read_data


# 整理因子
def arange_iv():        
    data0=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\wavelet_decomposition_with_Matlab\decomposition_result\multi_haar_3lev_predictor(hybrid_across_with_D1-D3、A1-A3).xlsx',index_col=None)
    data0=data0.iloc[6:]
    
    potential_iv_flow= pd.concat([pd.DataFrame(np.array(data0.iloc[5:-1,0:7]),columns=np.array(data0.iloc[:,0:7].columns)+np.array(7*['_t-1'])), \
                    pd.DataFrame(np.array(data0.iloc[4:-2,0:7]),columns=np.array(data0.iloc[:,0:7].columns)+np.array(7*['_t-2']))], axis=1)

    potential_iv_rain= pd.concat([pd.DataFrame(np.array(data0.iloc[5:-1,7:14]),columns=np.array(data0.iloc[:,7:14].columns)+np.array(7*['_t-1'])), \
                    pd.DataFrame(np.array(data0.iloc[4:-2,7:14]),columns=np.array(data0.iloc[:,7:14].columns)+np.array(7*['_t-2'])),\
                        pd.DataFrame(np.array(data0.iloc[3:-3,7:14]),columns=np.array(data0.iloc[:,7:14].columns)+np.array(7*['_t-3']))], axis=1)
    
    index_set=[]
    for i in range(6):
        index_set.append(pd.DataFrame(np.array(data0.iloc[5-i:-1-i,14:]),columns=np.array(data0.iloc[:,14:].columns)+np.array(140*['_t-'+str(i+1)])))
    index_set=pd.concat(index_set,axis=1)
    
    data1=pd.read_excel(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\wavelet_decomposition_with_Matlab\decomposition_result\multi_haar_3lev_predictant(hybrid_across_with_D1-D3、A1-A3).xlsx',index_col=None)
    data1=pd.DataFrame(np.array(data1.iloc[12:]),columns=data1.columns)
    
    potential_iv=[]
    for i in range(6):
        potential_iv.append(pd.concat([potential_iv_flow, potential_iv_rain, index_set, data1.iloc[:,i]],axis=1))
    
    return potential_iv

# 整理因子为csv文件
def transfer_2_csv():
    scaler = MinMaxScaler(feature_range=(-1, 1))
    
    save_file=['\potential_iv_multi_WDDFF1_d1(with_D1-D3、A1-A3).csv','\potential_iv_multi_WDDFF1_d2(with_D1-D3、A1-A3).csv','\potential_iv_multi_WDDFF1_d3(with_D1-D3、A1-A3).csv',\
                    '\potential_iv_multi_WDDFF1_a1(with_D1-D3、A1-A3).csv', '\potential_iv_multi_WDDFF1_a2(with_D1-D3、A1-A3).csv', '\potential_iv_multi_WDDFF1_a3(with_D1-D3、A1-A3).csv']
    csv_data=arange_iv()
    for i in range(len(csv_data)):
        csv_col=csv_data[i].columns
        csv_data_=scaler.fit_transform(csv_data[i])
        csv_data_=pd.DataFrame(csv_data_, columns=csv_col)
        csv_data_=pd.concat([pd.DataFrame(np.arange(len(csv_data_))+1,columns=['Id']), csv_data_],axis=1)
        csv_data_.to_csv(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\PMIS_for_IVS_with_R\csv'+save_file[i], index=False,sep=',')

if __name__ == "__main__":
    transfer_2_csv()