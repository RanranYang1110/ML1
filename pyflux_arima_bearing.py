#-*- coding:utf-8 -*-
# @author: qianli
# @file: pyflux_arima_bearing.py
# @time: 2019/06/14

import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
data1 = pd.read_csv('NU214.csv',header=None)
# data1 = pd.read_csv('6311.csv', header=None)
N = data1.shape[0]
percent = 0.1
len = int(N * percent)
RMSE = []
'''单一样本预测'''
# i = 2
# print(i)
# data = data1.loc[i * len + len - 49:i * len + len, 0]
# # data = data1.loc[i*len:i*len+len, 0]
# data = pd.DataFrame(data)
#
# model = pf.ARIMA(data=data, ar=4, ma=4, target=0, integ=1)
# x = model.fit(method='MLE')
#
# a = model.predict(h=30, intervals=False)
# a1 = data.shift(1)
#
# plt.close()
# plt.figure(figsize=(10, 5))
# plt.plot(a.index, a.iloc[:, 0])
# # plt.plot(data1.index[i * len + len:i * len + len + 30], data1.loc[i * len + len:i * len + len + 30, 0])
# plt.plot(data1.index[: i * len + len + 30], data1.loc[:i * len + len + 29, 0])
# name = '6311 ' + str(i * 10) + '% -- ' + str((i + 1) * 10) + '%'
# plt.title(name + '  RMSE: %.4f' % np.sqrt(
#     sum((a.iloc[:, 0] - data1.loc[i * len + len + 1:i * len + len + 30, 0]) ** 2) / a.size))
#
# plt.xlabel('samples')
# plt.ylabel('CV')
# plt.ylim([0, 1])
# savepath = r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演\fig\arima"
# figname = name + '.png'
# plt.savefig(os.path.join(savepath, figname), format='png', transparent=True, dpi=300)
# error = np.sqrt(sum((a.iloc[:, 0] - data1.loc[i * len + len + 1:i * len + len + 30, 0]) ** 2) / a.size)
#%%
for i in range(9):
    print(i)
    data = data1.loc[i*len+len-49:i*len+len, 0]
    # data = data1.loc[i*len:i*len+len, 0]
    data = pd.DataFrame(data)
    #
    # plt.figure(figsize=(15, 5))
    # plt.plot(data.index,data.loc[:,0])
    # plt.ylabel('CV')
    # plt.title('CV Data')

    model = pf.ARIMA(data=data, ar=10, ma=10, target=0, integ=0)
    x = model.fit(method='MLE')
    # x.summary()
    # model.plot_z(figsize=(15,5))
    # model.plot_fit(figsize=(15,10))
    # model.plot_predict_is(h=50, figsize=(15,5))
    # model.plot_predict(h=20,past_values=20,figsize=(15,5))
    a = model.predict(h=30, intervals=False)
    # plt.close()
    # plt.figure(figsize=(10,5))
    # plt.plot(a.index, a.iloc[:,0])
    # # plt.plot(data1.index[i * len + len:i * len + len + 30], data1.loc[i * len + len:i * len + len + 30, 0])
    # plt.plot(data1.index[: i * len + len + 30], data1.loc[:i * len + len + 29, 0])
    # name = '6311 ' + str(i*10) + '% -- ' + str((i+1)*10) + '%'
    # plt.title(name + '  RMSE: %.4f' % np.sqrt(sum((a.iloc[:, 0] - data1.loc[i * len + len + 1:i * len + len + 30, 0]) ** 2) / a.size))
    #
    # plt.xlabel('samples')
    # plt.ylabel('CV')
    # plt.ylim([0,1])
    # savepath = r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演\fig\arima"
    # figname = name + '.png'
    # plt.savefig(os.path.join(savepath, figname), format='png', transparent=True, dpi=300)
    error = np.sqrt(sum((a.iloc[:, 0] - data1.loc[i * len + len + 1:i * len + len + 30, 0]) ** 2) / a.size)

    RMSE.append(error)
np.array(RMSE)
#%%
# aa = model.predict_is(h=30,fit_once=True, fit_method='MLE')
