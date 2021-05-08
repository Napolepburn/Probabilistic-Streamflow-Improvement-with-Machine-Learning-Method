# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
#from sklearn.preprocessing import MinMaxScaler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 获取当前文件夹的上一级路径
import read_data


# 读取数据
Flow_data_monthly,_ =read_data.read_flow_data()
Rainfall_data_monthly,_ =read_data.read_rainfall_data()
climate_index=read_data.read_climate_index()

# 整理分析数据1（对齐数据, 1957-01~2008-12）
Flow_data_monthly1=Flow_data_monthly.loc['1957-01':'2008-12',:]
Rainfall_data_monthly1=Rainfall_data_monthly.loc['1957-01':'2008-12',:]
climate_index1=climate_index.loc['1957-01':'2008-12',:]

# 整理分析数据2（滞时数据, 留出12个月窗口以作分析）
Flow_data_monthly2=Flow_data_monthly1.iloc[12:]
Rainfall_data_monthly2=Rainfall_data_monthly1.iloc[12:]
climate_index2=climate_index1.iloc[12:]


# 整理DDFF预测框架的潜在因子
def arrange_potential_iv():
	potential_iv_flow=pd.DataFrame(np.vstack((Flow_data_monthly1.iloc[11:-1,].values.flatten(), \
					Flow_data_monthly1.iloc[10:-2,].values.flatten())).T, columns=['flow_t-1', 'flow_t-2'])

	potential_iv_rain=pd.DataFrame(np.vstack((Rainfall_data_monthly1.iloc[11:-1,].values.flatten(), \
					Rainfall_data_monthly1.iloc[10:-2,].values.flatten(),Rainfall_data_monthly1.iloc[9:-3,].values.flatten())).T, columns=['rain_t-1', 'rain_t-2', 'rain_t-3'])

	potential_index=[]
	potential_indexname=[]
	for j in range(len(climate_index1.columns)):
		for i in range(6):
			potential_indexname.append(climate_index1.columns[j]+'_t-'+str(i+1))
			potential_index.append(climate_index1.iloc[11-i:-1-i,j].values)
	potential_index=np.array(potential_index).T
	potential_iv_index=pd.DataFrame(potential_index, columns=potential_indexname)

	return pd.concat([potential_iv_flow, potential_iv_rain, potential_iv_index],axis=1)

# 整理潜在因子为csv文件
def transfer_2_csv():
	scaler = MinMaxScaler(feature_range=(-1, 1))
	
	csv_data=pd.concat([potential_iv, pd.DataFrame(Flow_data_monthly2.values.flatten(),columns=['target'])],axis=1)
	csv_col=csv_data.columns
	csv_data=scaler.fit_transform(csv_data)
	csv_data=pd.DataFrame(csv_data, columns=csv_col)
	csv_data=pd.concat([pd.DataFrame(np.arange(len(Flow_data_monthly2))+1,columns=['Id']), csv_data],axis=1)
	csv_data.to_csv(r'C:\Users\Administrator\Desktop\径流预报\实验\code\博罗月径流预测\PMIS_for_IVS_with_R\csv\potential_iv_DDFF1.csv',index=False,sep=',')

if __name__ == "__main__":
    potential_iv=arrange_potential_iv()
    transfer_2_csv()











