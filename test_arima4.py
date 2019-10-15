#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima4.py
# @time: 2019/06/14

import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
import itertools
import warnings

#引入数据
data = sm.datasets.co2.load()
#对引入的数据进行变形
index = pd.DatetimeIndex(start=data.data['date'][0].decode('utf-8'),
                             periods=len(data.data),freq='W-SAT')
co2 = pd.DataFrame(data.data['co2'], index=index, columns=['co2'])
print(co2.index)#检查co2的索引

y1 = co2["co2"].resample("MS").mean()#获得每个月的平均值
print(y1.isnull().sum)#5个 检测空白值
#处理数据中的缺失项
y1 = y1.fillna(y1.bfill())#填充缺失值
y = y1[:500]
'''初始数据可视化'''
plt.figure(figsize=(15, 6))
plt.title("原始数据", loc="center", fontsize=20)
plt.plot(y)

'''确定p、q'''
#找合适的p d q
#初始化 p d q
p = range(0,2)
d = range(0,2)
q = range(0,2)
print("p=",p,"d=",d,"q=",q)
#产生不同的pdq元组,得到 p d q 全排列
pdq=list(itertools.product(p,d,q))
print("pdq:\n",pdq)
seasonal_pdq=[(x[0],x[1],x[2],12) for x in pdq]
print('SQRIMAX:{} x {}'.format(pdq[1],seasonal_pdq[1]))
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(y,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

results.plot_diagnostics(figsize=(15, 12))
plt.show()
#%%
# #进行验证预测
pred=results.get_prediction(start=pd.to_datetime('1998-01-01'),dynamic=False)
pred_ci=pred.conf_int()
print("pred ci:\n",pred_ci)#获得的是一个预测范围，置信区间
print("pred:\n",pred)#为一个预测对象
print("pred mean:\n",pred.predicted_mean)#为预测的平均值

#进行绘制预测图像
ax=y['1990':].plot(label="observed")
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
y_truth=y['1998-01-01':]
mse=((y_forecasted-y_truth)**2).mean()
print("MSE:",mse)
#%%
# pred=results.forecast(10)
pred=results.get_prediction(start=pd.to_datetime('1999-11-01'),end=pd.to_datetime('2000-11-01'),dynamic=False)
pred_ci=pred.conf_int()
print("pred ci:\n",pred_ci)#获得的是一个预测范围，置信区间
print("pred:\n",pred)#为一个预测对象
print("pred mean:\n",pred.predicted_mean)#为预测的平均值

#进行绘制预测图像
ax=y['1990':].plot(label="observed")
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
y_truth=y1['1999-11-01':'2000-11-01']
mse=((y_forecasted-y_truth)**2).mean()
print("MSE:",mse)