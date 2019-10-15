#-*- coding:utf-8 -*-
# @author: qianli
# @file: com_gpr_arima.py
# @time: 2019/06/12


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
import warnings
warnings.filterwarnings('ignore')

class arima():
    def __init__(self, size):
        print('--calculate arima')
        self.size = size
    def draw_trend(self, timeSeries):
        # f = plt.figure(facecolor='white')
        # 对size个数据进行移动平均
        rol_mean = timeSeries.rolling(window=self.size).mean()
        # 对size个数据进行加权移动平均
        timeSeries.ewm(span=self.size).mean().plot(color='black', label='Weighted Rolling Mean')
        timeSeries.plot(color='blue', label='Original')
        rol_mean.plot(color='red', label='Rolling Mean')
        plt.legend(loc='best')
        plt.title('Rolling Mean')
        plt.show()

    def testStationarity(self, timeSeries):
        '''检验平稳性'''
        t = sm.tsa.stattools.adfuller(timeSeries)
        '''对上述函数求得的值进行语义描述'''

        output = pd.DataFrame(index=['Test Statistic Value', 'p-value', 'Lags Used', 'Number of Observations Used',
                                     'Critical Value(1%)', 'Critical Value(5%)', 'Critical Value(10%)'],
                              columns=['value'])
        output['value']['Test Statistic Value'] = t[0]
        output['value']['p-value'] = t[1]
        output['value']['Lags Used'] = t[2]
        output['value']['Number of Observations Used'] = t[3]
        output['value']['Critical Value(1%)'] = t[4]['1%']
        output['value']['Critical Value(5%)'] = t[4]['5%']
        output['value']['Critical Value(10%)'] = t[4]['10%']
        print(output)
        return output

    def draw_pc(self, timeSeries):
        plot_acf(timeSeries)  # 自相关图
        plot_pacf(timeSeries)  # 偏自相关图
        plt.show()

    def build_model(self, timeSeries, ts, predictSteps=30):
        (p, q) = (sm.tsa.arma_order_select_ic(timeSeries, max_ar=3, max_ma=3, ic='aic')['aic_min_order'])
        model = ARMA(timeSeries, order=(p, q)).fit(disp=-1, maxiter=100)
        # predict_data = model.predict(timeSeries)
        predict_data = model.forecast(30)
        predict_data = pd.Series(predict_data[0])
        pre_y = self.shift(predict_data, ts, 1)
        pre_y.dropna(inplace=True)
        pre_y.reset_index(drop=True)
        predict_data.reset_index(drop=True)
        # result_arma = model.fit(disp=-1, method='css')
        # predict_data = result_arma.predict()
        return predict_data, pre_y

    def dif(self, timeSeries, order):
        timeSeries = timeSeries.diff(order)
        timeSeries.dropna(inplace=True)
        return timeSeries

    def shift(self, pred, timeSeries, order):
        ts_shift =  timeSeries.shift(order)
        ts_shift.reset_index(drop=True)
        ts_recorver = pred.add(ts_shift)
        return ts_recorver
    # def recover_predict(self, pre_value):

    def plt_result(self, timeSeries, pred):
        plt.figure(facecolor='white')
        pred.plot(color='blue', label = 'Predict')
        timeSeries.plot(color='red', label = 'Original')
        plt.legend(loc='best')
        pred = np.array(pred)
        timeSeries = np.array(timeSeries)
        plt.title('RMSE: %.4f' % np.sqrt(sum((pred[1:] - timeSeries[1:]) ** 2) / timeSeries.size))
        plt.show()

    def cal_rmse(self, timeSeries, pred):
        pred = np.array(pred)
        timeSeries = np.array(timeSeries)
        return np.sqrt(sum((pred - timeSeries[:]) ** 2) / (timeSeries.size-1))

def shift1(ts, pre_y, order=1):
    ts_shift = ts.shift(order)
    ts_shift.reset_index(drop=True)
    pre_y.reset_index(drop=True)
    ts_recorver = pre_y.add(ts_shift)
    ts_recorver.dropna(inplace=True)
    return ts_recorver

#%%
import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")
# data = pd.read_csv('NU214.csv',header=None)
data = pd.read_csv('6311.csv', header=None)
data = data.loc[:,0]
N = data.shape[0]
percent = 0.1
trainsteps = 30
predictSteps = 30
data_select = data#data[0 : int( N * percent)]
i = 0
RMSE = []
len = int(N * percent)

AR = arima(size=10)
pp = []
qq = []
for i in range(0, 9):#N-30):

    print(i)
    # ts = data_select[i:i+trainsteps]
    # real_y = data_select[i+trainsteps:i+trainsteps+30].reset_index(drop=True)
    ts = data_select[i*len+len-30:i*len+len]
    real_y = data_select[i*len+len :i*len+len + predictSteps]
    ts_diff = AR.dif(ts, order=1)
    t = AR.testStationarity(ts_diff)

    (p, q) = (sm.tsa.arma_order_select_ic(ts_diff, max_ar=4, max_ma=4, ic='aic')['aic_min_order'])
    try:
        model = ARMA(ts_diff, order=(p, q)).fit(disp=-1, maxiter=100)
    except:
        model = ARMA(ts_diff, order=(p, 1)).fit(disp=-1, maxiter=100)
    pp.append(p)
    qq.append(q)
    # predict_data = model.predict(timeSeries)
    predict_data = model.forecast(30)
    predict_data = pd.Series(predict_data[0])
    ts_shift = ts.shift(1)

    ts_shift = ts_shift.reset_index(drop=True)
    pre_y = predict_data.add(ts_shift)
    # ts_recorver = pre_y.add(ts_shift)
    # pre_y = pre_y.dropna(inplace=True)
    pre_y = pre_y.reset_index(drop=True)
    real_y = real_y.reset_index(drop=True)
    # pre_y = shift1(ts, predict_data, 1)
    # pre_y.dropna(inplace=True)
    # pre_y.reset_index(drop=True)
    # predict_data.reset_index(drop=True)
    error = np.sqrt(sum((pre_y[1:] - real_y[1:]) ** 2) / (real_y.size))
    RMSE.append(error)
    # AR.plt_result(real_y, pre_y)
        # t = AR.testStationarity(ts_diff)
        # predict_data, pre_y = AR.build_model(ts_diff, ts=ts)
        # error = AR.cal_rmse(timeSeries=real_y, pred=pre_y)
        # RMSE.append(error)
# data2 = model.dif(data_select, order=1)
# ts = model.testStationarity(data2)
