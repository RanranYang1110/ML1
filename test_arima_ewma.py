#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_arima_ewma.py
# @time: 2019/06/14

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pyflux as pf
from statsmodels.graphics.tsaplots import acf, pacf, plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_model import ARIMA
import os
os.chdir(r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演")

def draw_ewma(data, size):
    f = plt.figure(facecolor='white')
    data1 = data.ewm(span=size).mean()
    plt.plot(data1, color='red', label='EWMA')
    # data1.plot(color='black', label='Weighted Rolling Mean')
    plt.plot(data, color='blue', label='Original')
    # data.plot(color='blue', label='Original')
    plt.legend(loc='best')
    plt.title('Rooling Mean')
    plt.show()
    return data1
def testStationarity(ts):
    t = sm.tsa.stattools.adfuller(ts) #adf检验
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
    # print(output)
    return output
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

def plot_predict(real_y, pre_y, rmse, i ):
    plt.close()
    ax = plt.figure(figsize=(8, 5))
    ax = real_y.plot(label="observed")
    # # data1 = data.loc[:1000, 0]
    pre_y.plot(ax=ax, label="static ForCast", alpha=.7, color='red', linewidth=5)
    # 在某个范围内进行填充
    plt.title('RMSE: %.4f' % rmse)
    ax.set_xlabel('number')
    ax.set_ylabel('CV')
    plt.ylim([0,1])
    plt.legend()
    plt.show()
    name = '6311 ' + str(i * 10) + '% -- ' + str((i + 1) * 10) + '%'
    # savepath = r"D:\1-轨交行业\5-中车永济轴承PHM\2-轴承寿命预测\11-验收相关\01 杭州寿命实验预演\fig\arima2"
    # figname = name + '.png'
    # plt.savefig(os.path.join(savepath, figname), format='png', transparent=True, dpi=300)

def cal_rmse(real_y, pre_y):
    rmse = np.sqrt(((real_y - pre_y) ** 2).mean())
    return rmse
#%%
def train1():
    ts = pd.read_csv('NU214.csv',header=None)
    time_series1 = ts
    ts_ewma = draw_ewma(ts, size=20)
    time_series1 = ts_ewma.loc[:,0]
    output = testStationarity(time_series1)
    print(output)

    N = ts.shape[0]
    percent = 0.1
    len = int(N * percent)
    RMSE = []
    for i in range(9):
        print(i)
        data = ts.loc[ i * len + len - 100: i * len + len, 0]
        (p, q) = (sm.tsa.arma_order_select_ic(data, max_ar=4, max_ma=4, ic='aic')['aic_min_order'])
        arma_model = sm.tsa.ARMA(time_series1, (0, 1)).fit(disp=-1, maxiter=100)
        # predict_data = arma_model.predict(start=str(1979), end=str(2018), dynamic=False)
        pre_y = arma_model.forecast(30)
        pre_y = pd.Series(pre_y[0])

        pre_y.index = pd.Index(range(i * len + len, i * len + len + 30))
        # pre_model, pre_y, pred_ci= predict(model, i * len + len, i * len + len + 29)
        real_data = ts.loc[i * len + len : i * len + len + 30, 0]
        draw_data = ts.loc[: i * len + len + 30, 0]
        plot_predict(draw_data, pre_y)
        rmse = cal_rmse(real_data, pre_y)
        RMSE.append(rmse)

def train2():
    ts = pd.read_csv('NU214.csv', header=None)
    ts_ewma = draw_ewma(ts, size=20)
    time_series1 = ts_ewma.loc[:, 0]
    output = testStationarity(time_series1)
    print(output)

    N = ts.shape[0]
    percent = 0.1
    len = int(N * percent)
    RMSE = []
    for i in range(9):
        print(i)
        data = ts.loc[i*len+len-49:i*len+len, 0]
        # data = data1.loc[i*len:i*len+len, 0]
        data = pd.DataFrame(data)
        #
        # plt.figure(figsize=(15, 5))
        # plt.plot(data.index,data.loc[:,0])
        # plt.ylabel('CV')
        # plt.title('CV Data')

        model = pf.ARIMA(data=data, ar=4, ma=4, target=0, integ=1)
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
        error = np.sqrt(sum((a.iloc[:, 0] - ts.loc[i * len + len + 1:i * len + len + 30, 0]) ** 2) / a.size)
        RMSE.append(error)
    return np.array(RMSE)

def train3():
    # ts = pd.read_csv('NU214.csv',header=None)
    ts = pd.read_csv('6311.csv',header=None)
    N = ts.shape[0]
    percent = 0.1
    len = int(N * percent)
    RMSE = []
    PP = []
    QQ = []
    for i in range(1):
        print(i)
        data = ts.loc[ i * len + len - 100: i * len + len - 1 , 0]
        data1 = data.diff(1).dropna()
        (p, q) = (sm.tsa.arma_order_select_ic(data1, max_ar=4, max_ma=4, ic='aic')['aic_min_order'])
        PP.append(p)
        QQ.append(q)
        arma_model = sm.tsa.ARMA(data1, (0, 0)).fit(disp=-1, maxiter=100)
        # predict_data = arma_model.predict(start=str(1979), end=str(2018), dynamic=False)
        # pre_y = arma_model.predict(start=30, end = i* len + len + 30)
        pre_y = arma_model.forecast(30)[0]
        pre_y = pd.Series(pre_y)
        pre_y.index = pd.Index(range(i * len + len, i * len + len + 30))
        diff_restored = pd.Series([data.values[0]], index=[data.index[0]]).append(data1).cumsum()

        start = i * len + len
        end = i * len + len + 30
        y = ts.loc[i * len + len - 100: i * len + len -1, 0]
        for j in range(start, end):
            y1 = y[j -1] + pre_y[j]
            y = pd.concat([y, pd.Series([y1],index=[j])])
        # pre_model, pre_y, pred_ci= predict(model, i * len + len, i * len + len + 29)
        predict_y = y[-30:]
        real_data = ts.loc[i * len + len : i * len + len + 29, 0]
        draw_data = ts.loc[: i * len + len + 30, 0]

        rmse = cal_rmse(real_data, predict_y)
        plot_predict(draw_data, predict_y, rmse, i)
        RMSE.append(rmse)
    return RMSE, PP, QQ

RMSE,PP,QQ = train3()
