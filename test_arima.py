#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima.py
# @time: 2019/06/12


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
time_series = pd.Series([151.0, 188.46, 199.38, 219.75, 241.55, 262.58, 328.22, 396.26, 442.04, 517.77, 626.52, 717.08,
                         824.38, 913.38, 1088.39, 1325.83, 1700.92, 2109.38, 2499.77, 2856.47, 3114.02, 3229.29, 3545.39,
                         3880.53, 4212.82, 4757.45, 5633.24, 6590.19, 7617.47, 9333.4, 11328.92, 12961.1, 15967.61])
time_series.index = pd.Index(sm.tsa.datetools.dates_from_range('1978', '2010'))
time_series.plot(figsize=(12,8))
plt.show()
#%%
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
time_series2 = time_series1.diff(1)
time_series2 = time_series2.dropna(how=any)
time_series2.plot(figsize=(8,6))
plt.show()

t = sm.tsa.stattools.adfuller(time_series2)
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

'''确定自相关系数和平均移动系数（p,q）'''
# 根据时间序列的识别规则，采用ACF图、PAC图、AIC准则和BIC准则（贝叶斯准则）
# 相结合的方式来确定ARMA模型的阶数，应当选取AIC和BIC值达到最小的那一组为理想阶数
plot_acf(time_series2) #自相关图
plot_pacf(time_series2) #偏自相关图
plt.show()

# r, rac, Q = sm.tsa.acf(time_series, qstat=True)
# prac = pacf(time_series, method='ywmle')
# table_data = np.c_[range(1, len(r)), r[1:], rac, prac[1:len(rac)+1], Q]
# table = pd.DataFrame(table_data, columns=['lag','AC','Q','PAC','Prob(>Q)'])
# print(table)

'''取值，并对模型进行估计'''
p, d, q = (0, 1, 1)
arma_mod = ARMA(time_series2, (p,d,q)).fit(disp=-1, method='mle')
summary = (arma_mod.summary2(alpha=0.05, float_format='%.8f'))
print(summary)

'''自动定阶'''
# (p, q) =(sm.tsa.arma_order_select_ic(time_series,max_ar=3,max_ma=3,ic='aic')['aic_min_order'])
# print(p)
# print(q)

'''残差和白噪声检验'''
arma_mod = ARMA(time_series2, (0,1,1)).fit(disp=-1, method='mle')
resid = arma_mod.resid
t = sm.tsa.stattools.adfuller(resid)
t = sm.tsa.stattools.adfuller(time_series2)
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

'''模型预测'''
arma_model = sm.tsa.ARMA(time_series2,(0,1)).fit(disp=-1, maxiter=100)
predict_data = arma_model.predict(start=str(1979), end=str(2013), dynamic=False)

from statsmodels.tsa.arima_model import ARIMA
time_series1.plot()
results = ARIMA(time_series1, (0,1,1)).fit(disp=-1, maxiter=100)
predict_data = results.predict(start=str(2010), end=str(2020), dynamic=False)
predict_data.plot(color='blue', label='Predict')

# arima = ARIMA(endog=logNums, order=(p, d, q))
# proArima = ARIMA(time_series1, (0,1,1)).fit(disp=-1, maxiter=100)
# fittedArima = proArima.fittedvalues.cumsum() + time_series1[0]
# fittedNums = np.exp(fittedArima)
# plt.plot(passengersNums, 'g-', lw=2, label=u'orignal')
# plt.plot(fittedNums, 'r-', lw=2, label=u'fitted')
# plt.legend(loc='best')
# plt.show()
'''预测结果还原'''
#对预测出来的数据，进行逆差分操作（由原始数据取对数后的数据加上预测出来的数据），然后再取指数即可还原
# time_series = time_series.diff(-1)
# time_series = np.exp(time_series)
# from statsmodels.tsa.arima_model import ARMA
# model = ARMA(diff_12, order=(1,1))
# result_arma = model.fit(disp=-1, method='css')

'''样本拟合'''
# predict_ts = predict_data.predict()
diff_shift_ts = time_series1.shift(1)
diff_recover_1 = predict_data.add(diff_shift_ts)
log_recover = np.exp(diff_recover_1)
log_recover.dropna(inplace=True)

ts = time_series[log_recover.index]  # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts[:])**2)/ts.size))
plt.show()
#%%
'''test_stationarity的统计性检验模块'''
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

'''移动平均图'''
def draw_trend(timeSeries, size):
    # f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    timeSeries.ewm(span=size).mean().plot(color='black', label='Weighted Rolling Mean')
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
    # rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')

def draw_ts(timeSeries):
    f = plt.figure(facecolor='white')
    timeSeries.plot(color='blue')
    plt.show()
# def test_stationarity(timeseries):
#     #Determing rolling statistics
#     rolmean =  timeseries.rolling(window=60).mean()
#     rolstd = timeseries.rolling(window=60).std()

def testStationarity(ts):
    t = adfuller(ts) #adf检验
    '''对上述函数求得的值进行语义描述'''

    output = pd.DataFrame(index=['Test Statistic Value', 'p-value', 'Lags Used', 'Number of Observations Used',
                                 'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)'], columns=['value'])
    # test statistic value
    # p-value 接受原假设的概率，p-value值越小越好
    # lags-used 延迟时间

    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)
    return output

def draw_acf_pacf(ts, lags=31):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts, lags=31, ax=ax1)
    ax2 = f.add_subplot(212)
    plot_pacf(ts, lags=31, ax=ax2)
    plt.show()

'''平稳性处理--对数变换'''
ts_log = np.log(time_series)
testStationarity(ts_log)
# draw_ts(ts_log)

'''平稳性处理--平滑法'''
# 移动平均法和指数平均法
# draw_trend(ts_log, 12)
# draw_trend(ts_log, 20)

'''平稳性处理--差分法'''
diff_12 = ts_log.diff(1)
diff_12.dropna(inplace=True)
diff_12_1 = diff_12.diff(1)
diff_12_1.dropna(inplace=True)
testStationarity(diff_12_1)

'''分解方法'''
from statsmodels.tsa.seasonal import seasonal_decompose
# 将时序数据分离成不同的长期趋势、季节趋势和随机成分。
decomposition = seasonal_decompose(time_series, model='additive') #加法模型，乘法模型model='multiplicative'
trend = decomposition.trend
sesonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(4,1,1)
plt.plot(time_series)
plt.subplot(4,1,2)
plt.plot(trend)
plt.subplot(4,1,3)
plt.plot(sesonal)
plt.subplot(4,1,4)
plt.plot(residual)

'''模式识别'''
'''通过一级差分结果建模'''
'''确定自相关系数和平均移动系数（p,q）'''
# 根据时间序列的识别规则，采用ACF图、PAC图、AIC准则和BIC准则（贝叶斯准则）
# 相结合的方式来确定ARMA模型的阶数，应当选取AIC和BIC值达到最小的那一组为理想阶数
log_acf = acf(time_series2, nlags=30) #nlags 往后延期30个值
log_pacf = pacf(time_series2, nlags=30, method='ols')
plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.plot(log_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(time_series2)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(time_series2)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(log_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(time_series2)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(time_series2)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')

plot_acf(time_series2) #自相关图
plot_pacf(time_series2) #偏自相关图
plt.show()
#%%

# 建立ARIMA模型
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
# model = ARMA(time_series2, order=(0,2))
# model = ARMA(time_series1, order=(0,1,2))
# result_arma = model.fit(disp=-1, method='css')
# model = ARMA(time_series2, order=(1,1))
#
# results = model.fit()

results = ARIMA(time_series2, (0,1,1)).fit(disp=-1, maxiter=100)
plt.figure(figsize=(12,7))
plt.plot(time_series2, color='blue')
plt.plot(results.fittedvalues, color='red')
plt.title('RSS: %.4f'% sum((results.fittedvalues-time_series2) ** 2))
outputs = results.forecast(5)
yhat = outputs[0]
'''样本拟合'''
predict_ts = results.predict()
diff_shift_ts = diff_12.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
log_recover = np.exp(diff_recover_1)
log_recover.dropna(inplace=True)



ts = time_series[log_recover.index]  # 过滤没有预测的记录
plt.figure(facecolor='white')
log_recover.plot(color='blue', label='Predict')
ts.plot(color='red', label='Original')
plt.legend(loc='best')
plt.title('RMSE: %.4f'% np.sqrt(sum((log_recover-ts)**2)/ts.size))
plt.show()