#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima5_bearing_pcl.py
# @time: 2019/06/14
#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima5.py
# @time: 2019/06/14
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

import numpy as np

import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
data = pd.read_csv('NU214.csv',header=None)
data1 = data.loc[:1000, 0]

'''确定p、q'''
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

mod = sm.tsa.statespace.SARIMAX(data1,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()
print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

# #进行验证预测
pred=results.get_prediction(start=1000, end=1029, dynamic=False)
pred_ci=pred.conf_int()
print("pred ci:\n",pred_ci)#获得的是一个预测范围，置信区间
print("pred:\n",pred)#为一个预测对象
print("pred mean:\n",pred.predicted_mean)#为预测的平均值

#进行绘制预测图像

ax = data.loc[1000:].plot(label="observed")
# data1 = data.loc[:1000, 0]
pred.predicted_mean.plot(ax=ax,label="static ForCast",alpha=.7,color='red',linewidth=5)
#在某个范围内进行填充
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

# 求取 MSE（均方误差）
y_forecasted=pred.predicted_mean
y_truth=data.values[1000:1030]
y_forecasted = y_forecasted.values
mse=((y_forecasted-y_truth)**2).mean()
rmse = np.sqrt(mse)
print('RMSE', rmse)
