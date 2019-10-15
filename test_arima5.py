#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima5.py
# @time: 2019/06/14
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import itertools
import warnings
import numpy as np


import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
data1 = pd.read_csv('NU214.csv',header=None)
# data1 = data.loc[:1000, 0]

N = data1.shape[0]
percent = 0.1
len = int(N * percent)
RMSE = []
def train_model(traindata):
    traindata = pd.DataFrame(traindata)
    mod = sm.tsa.statespace.SARIMAX(traindata, order=(0, 1, 1), seasonal_order=(1, 1, 1, 12),
                                    enforce_stationarity=False, enforce_invertibility=False)

    # mod = sm.tsa.statespace.SARIMAX(traindata, order=(1, 1, 1), seasonal_order=(1,1,0,1),
    #                                 enforce_stationarity=False, enforce_invertibility=False)
    results = mod.fit()
    # print(results.summary().tables[1])
    # results.plot_diagnostics(figsize=(15, 12))
    # plt.show()
    return results

def predict(testdata, start, end):
    pred = testdata.get_prediction(start=start, end=end, dynamic=False)
    pre_y = pred.predicted_mean
    pre_y.index = pd.Index(range(start, end+1))
    pred_ci = pred.conf_int()
    pred_ci.index = pd.Index(range(start, end+1))
    return pred, pre_y, pred_ci

def plot_predict(real_y, pred_ci, pre_y ):

    ax = plt.figure(figsize=(8, 5))
    ax = real_y.plot(label="observed")
    # # data1 = data.loc[:1000, 0]
    pre_y.plot(ax=ax, label="static ForCast", alpha=.7, color='red', linewidth=5)
    # 在某个范围内进行填充
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.2)
    ax.set_xlabel('number')
    ax.set_ylabel('CV')
    plt.ylim([0,1])
    plt.legend()
    plt.show()

def cal_rmse(real_y, pre_y):
    rmse = np.sqrt(((real_y - pre_y) ** 2).mean())
    return rmse
#%%
RMSE = []
for i in range(9):
    print(i)
    data = data1.loc[ i * len + len - 100: i * len + len, 0]
    model = train_model(data)
    pre_model, pre_y, pred_ci= predict(model, i * len + len, i * len + len + 29)
    real_data = data1.loc[i * len + len : i * len + len + 30, 0]
    draw_data = data1.loc[: i * len + len + 30, 0]
    plot_predict(draw_data, pred_ci, pre_y)
    rmse = cal_rmse(real_data, pre_y)
    RMSE.append(rmse)
# '''确定p、q'''
#找合适的p d q
#初始化 p d q
# p = range(0,2)
# d = range(0,2)
# q = range(0,2)
# print("p=",p,"d=",d,"q=",q)
# #产生不同的pdq元组,得到 p d q 全排列
# pdq=list(itertools.product(p,d,q))
# print("pdq:\n",pdq)
# seasonal_pdq=[(x[0],x[1],x[2],12) for x in pdq]
# print('SQRIMAX:{} x {}'.format(pdq[1],seasonal_pdq[1]))
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(data1,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit()
#
#             print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
#         except:
#             continue
#%%
mod = sm.tsa.statespace.SARIMAX(data1,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

