#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima2.py
# @time: 2019/06/14
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA

import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
data = pd.read_csv('NU214.csv',header=None)
data1 = data.loc[:1000, 0]
#%%
# 定阶 一般阶数不超过length/10
pmax = 3#int(len(data1) / 100)
qmax = 3#int(len(data1) / 100)

#bic矩阵
bic_matrix = []
for p in range(pmax + 1):
    tmp = []
    for q in range(qmax + 1):
        try:
            tmp.append(ARIMA(data1, (p, 1, q)).fit().bic)
        except:
            tmp.append(None)
bic_matrix.append(tmp)
#从中可以找到最小值
bic_matrix = pd.DataFrame(bic_matrix)

#先用stack展平，然后用idxmin找出最小值位置
# p, q = bic_matrix.stack().idxmin()
p = 0
q = 1
model = ARIMA(data, (p, 1, q)).fit()
model.summary2()

pre_y = model.forecast(30)
pre_y = pre_y[0].reshape([-1,1])
plt.plot(pre_y)
data333 = data.values[1000:1030].reshape([-1,1])
plt.plot(data333[1000:1030])

np.sqrt((np.sum(data333 - pre_y) ** 2) / 30)