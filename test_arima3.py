#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima3.py
# @time: 2019/06/14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
time_series = pd.Series([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 396.26, 442.04, 517.77, 626.52, 717.08,
                         824.38, 913.38, 1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 3114.02, 3229.29, 3545.39,
                         3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61])
ddd = pd.Series([3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61])
ddd.index = pd.Index(sm.tsa.datetools.dates_from_range('2001', '2010'))
time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1978', '2010'))
time_series.plot(figsize=(12,8))
ddd.plot(figsize=(12,8))
plt.show()

'''取对数，将其转化为线性趋势'''
time_series1 = np.log(time_series)
time_series1.plot(figsize=(8,6))
plt.show()
#%%
'''adf稳定性检验'''
t = sm.tsa.stattools.adfuller(time_series1, )
output = pd.DataFrame(index=['Test Statistic Value', 'p-value', 'Lags Used', 'Number of Observations Used',
                             'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)'], columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print(output)
'''t统计量若低于任何置信度间的临界值，则判断序列平稳'''

'''对序列进行差分处理'''
time_series2 = time_series1.diff(1).dropna()
# time_series2 = time_series2.dropna(how=any)
# time_series2.plot(figsize=(8,6))
# plt.show()

# '''确定自相关系数和平均移动系数（p,q）'''
# plot_acf(time_series2) #自相关图
# plot_pacf(time_series2) #偏自相关图
# plt.show()


'''自动定阶'''
(p, q) =(sm.tsa.arma_order_select_ic(time_series2,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
print(p)
print(q)



'''模型预测'''
# arma_model = sm.tsa.ARMA(time_series2,(0,1)).fit(disp=-1, maxiter=100)
# predict_data = arma_model.predict(start=str(1979), end=str(2013), dynamic=False)
arma_model = sm.tsa.ARMA(time_series1,(0, 1)).fit(disp=-1, maxiter=100)
predict_data = arma_model.predict(start=str(1979), end=str(2018), dynamic=False)

# arma_model = sm.tsa.ARIMA(time_series1,(0,1, 1)).fit(disp=-1, maxiter=100)
# predict_data = arma_model.predict(start=str(1979), end=str(2018), dynamic=False, )
'''预测结果还原'''
diff_restored = pd.Series([time_series1[0]], index=[time_series1.index[0]]) .append(time_series2).cumsum()
exp_restored = np.exp(diff_restored)
recover_ts = exp_restored + predict_data
#%%
'''样本拟合'''
diff_shift_ts = time_series1.shift(33)
diff_recover_1 = predict_data.add(diff_shift_ts)
log_recover = np.exp(diff_recover_1)
log_recover.dropna(inplace=False)

ts = time_series[log_recover.index]  # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts[:])**2)/ts.size))
plt.show()
