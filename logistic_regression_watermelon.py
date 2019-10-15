#-*- coding:utf-8 -*-
# @author: qianli
# @file: logistic_regression_watermelon.py
# @time: 2019/03/20
#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"D:\7-学习资料\a-study\machinlearning\daima")
data = pd.read_csv('watermelon.csv')
dataset = data.iloc[:,1:3].values
#%%
pos = data[data['好瓜'] == '是'].index.tolist()
df2 = pd.DataFrame(np.concatenate([np.ones([8,1]),np.zeros([8,1])]),columns=['labels'])
data['labels'] = df2['labels'].apply(int)
# y = data['labels']
# dataset1 = dataset[dataset[:,0].argsort()] #按照密度大小值进行排序
#
# '''查看原始数据'''
# plt.plot(dataset1[:,0],dataset1[:,1])
# plt.show()
'''查看数据集'''
X = dataset
y = data['labels']
m, n = np.shape(X)
f1 = plt.figure(1)
plt.title('watermelon_3a')
plt.xlabel('density')
plt.ylabel('ratio_sugar')
plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'k', s=100, label = 'bad')
plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'good')
plt.legend(loc = 'upper right')

'''逻辑回归'''
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y,test_size=0.4, random_state=0)

