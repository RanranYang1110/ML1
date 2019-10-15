#-*- coding:utf-8 -*-
# @author: qianli
# @file: DecisionTree_ID3.py
# @time: 2019/03/22
#%%
import numpy as np
def Entropy(p):
    return - p * np.log2(p)

from sklearn.datasets import load_iris
trainData = [[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],
             [1,0,0,0],[1,0,0,1],[1,1,1,1],[1,0,1,2],[1,0,1,2],
             [2,0,1,2],[2,0,1,1],[2,1,0,1],[2,1,0,2],[2,0,0,0]]
trainLabel = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
num_class = np.unique(trainLabel)
num_feature = []
pos = []
for i in range(len(num_class)):
    pos1 = np.where(trainLabel == num_class[i])[0]
    pos.append(list(pos1))
    num_feature.append(len(pos1))
#%%
from collections import Counter
import pandas as pd
def calculate_Entropy(dataset):
    num = pd.Series(Counter(dataset)) #dataset的不重复数值及个数
    len_dataset = len(dataset)
    E = 0
    for i in range(num.shape[0]):
        p1 = num[i] / len_dataset
        E += - p1 * np.log2(p1)
    return E
def information_Gain(X, y):
    num_feature, num_class = X.shape
    E1 = []
    for i in range(num_feature):
        data = X[:, i]
        num = pd.Series(Counter(data))  # dataset的不重复数值及个数
        len_dataset = num_feature

        E1.append(calculate_Entropy(X[:,i]))


# def calculate_Entropy(X,y):
#     n = len(y)
#     all_n = sum(y)
#     E = 0
#     for i in range(n):
#         E += y[i] / all_n * np.log2(y[i]/all_n)
#     num_train, num_features = X.shape
#     for i in range(num_feature):
#         trainData_1 = trainData[:,i]
#     return E