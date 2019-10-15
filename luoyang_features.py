#-*- coding:utf-8 -*-
# @author: qianli
# @file: luoyang_features.py
# @time: 2019/04/15

import pandas as pd
import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\10-洛阳故障实验\data")
filedir = os.listdir()
faultLevel_NU214 = ['严重','严重','严重','中等','严重','严重','轻微','轻微','轻微','轻微',
                    '轻微','轻微','中等','严重','严重','轻微','中等','轻微','中等','轻微',
                    '轻微','轻微','轻微','中等','轻微','中等','轻微','轻微','轻微','轻微']
faultPos_NU214 = ['内圈','内圈','内圈','内圈','滚子','滚子','保持架','滚子','外圈','内圈',
                  '保持架','内圈','内圈','内圈','内圈','滚子','滚子','滚子','滚子','保持架',
                  '保持架','保持架','外圈','保持架','外圈','保持架','外圈','外圈','保持架','外圈']
faultLevel_6311 = ['严重','严重','严重','严重','严重','严重','轻微','轻微','轻微','轻微',
                   '轻微','轻微','轻微','中等','严重','轻微','轻微','中等','轻微','轻微',
                   '轻微','轻微','轻微','轻微','轻微','中等','中等','轻微','轻微','轻微']
faultPos_6311 = ['外圈','外圈','内圈','外圈','内圈','内圈','滚子','内圈','外圈','保持架',
            '保持架','内圈','内圈','内圈','内圈','外圈','外圈','外圈','外圈','保持架',
            '保持架','保持架','滚子','保持架','滚子','滚子','保持架','滚子','滚子','滚子']
#%%
df = pd.DataFrame()
for i in range(30):
    with open(filedir[i]) as f:
        data1 = pd.read_csv(f)
    data1['test'] = '第' + filedir[i][16:-4] + '组'
    data1['6311严重程度'] = faultLevel_6311[i]
    data1['6311故障位置'] = faultPos_6311[i]
    data1['NU214严重程度'] = faultLevel_NU214[i]
    data1['NU214故障位置'] = faultPos_NU214[i]
    df = pd.concat([df, data1])
df.to_csv("AutofeatureTable.csv",encoding='utf_8_sig')
#%%
# with open(filedir[2]) as f:
#     data2 = pd.read_csv(f)
#
# df5 = pd.concat([data1,data2])
# df5.head()
pos = df['speed>2000','6311严重程度'=='严重']
#%%

import numpy as np
data = np.array([1,2,3,4])
np.sqrt(data)
np.sqrt((data, data))

data1 = [1,2,3,4]
np.sqrt(data1)
np.sqrt((data1,data1))
np.sqrt([data1, data1])
