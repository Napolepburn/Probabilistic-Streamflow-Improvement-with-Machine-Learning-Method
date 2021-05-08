# -*- coding: utf-8 -*

import numpy as np
import pandas as pd
import os
import matlab
import matlab.engine
engine = matlab.engine.start_matlab()

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # 添加文件所在文件夹
import IVS_PMI_output_csv_for_DDFF_input

# 整理DDFF预测框架的潜在因子
potential_iv=IVS_PMI_output_csv_for_DDFF_input.arrange_potential_iv()
potential_iv=pd.concat([potential_iv, pd.DataFrame(IVS_PMI_output_csv_for_DDFF_input.Flow_data_monthly2.values.flatten(),columns=['target'])],axis=1)

# 经PMI排序的因子
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

# 计算top20排序因子的MI相关
MI_value1=[]
for i in range(20):
	MI_value1.append(engine.kernelmi(matlab.double(potential_iv[PMIS_iv[i]].values.tolist()),matlab.double(potential_iv.iloc[:,-1].values.tolist())))

print(MI_value1)