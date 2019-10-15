#-*- coding:utf-8 -*-
# @author: qianli
# @file: pyflux_arima.py
# @time: 2019/06/14

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
# data = pd.read_csv('NU214.csv',header=None)
data = pd.read_csv('https://vincentarelbundock.github.io/Rdatasets/csv/datasets/sunspot.year.csv')
data.index = data['time'].values
# data = pd.DataFrame([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 396.26, 442.04, 517.77, 626.52, 717.08,
#                      824.38, 913.38, 1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 3114.02, 3229.29, 3545.39], columns=['value'])
#                          # 3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61], columns=['value'])
# data.index = pd.Index(sm.tsa.datetools.dates_from_range('1978', '2000'))
# data1 = data.loc[:,0]
# N = data1.shape[0]
plt.figure(figsize=(15, 5))
# data.plot(color='blue', label='Original')
plt.plot(data.index,data['value'])
plt.ylabel('Sunspots')
plt.title('Yearly Sunspot Data')

model = pf.ARIMA(data=data, ar=4, ma=4, target='value', integ=1)
x = model.fit(method='MLE')
x.summary()
# model.plot_z(figsize=(15,5))
# model.plot_fit(figsize=(15,10))
# model.plot_predict_is(h=50, figsize=(15,5))
# model.plot_predict(h=20,past_values=20,figsize=(15,5))
a = model.predict(h=10, intervals=False)

mm = [3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61]
plt.figure(facecolor='white')
plt.plot(a.values)
plt.plot(mm)
plt.ylim([0,16000])
# a.plot(color='blue', label='Predict')
# data.plot(color='red', label='Original')
# plt.legend(loc='best')
# plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts[:])**2)/ts.size))
plt.show()