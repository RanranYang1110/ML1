#-*- coding:utf-8 -*-
# @author: qianli
# @file: soybean_arima.py
# @time: 2019/07/30
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
plt.style.use('ggplot')
import os
os.chdir(r"D:\7-学习资料\a-study\machinlearning\万门大学课件\课件（四）\0818\下午\大豆价格预测\ARIMA")

data = pd.read_csv('soybean_price_guangdong.csv', parse_dates=['date'], index_col=['date'])
data.index = pd.to_datetime(data.index)
#%%
data.head()
data.plot()
plt.ylabel('price')
data.info()
#%%
ts = data['price']
ts.head()
ts.head().index