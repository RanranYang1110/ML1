#-*- coding:utf-8 -*-
# @author: qianli
# @file: sk_kmeans.py
# @time: 2019/04/10

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
data = iris.data[:100,:4]
labels = iris.target[:100]
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)
model = KMeans()
# model.fit(X_train,3)
# k = 3
# np.random.randint(0,len(X_train))
#%%
# class kmeans(object):
#     def __init__(self):
#         self.k = 3
#         self.p = 3
#         print('初始化成功')
#     def pdistCal(self, x, y):
#         x = np.array(x).reshape(1,-1)
#         y = np.array(y).reshape(1,-1)
#         '''计算距离'''
#         dis1 = np.sum((x - y) ** self.p) ** (1/self.p)
#         return dis1
#     def MiuCal(self, x):
#         '''计算均值向量'''
#         x = np.array(x)
#         u = np.mean(x, axis=0)
#         return u
#     def classCu(self, datasets, miuS):
#         '''计算均值向量
#         datasets:数据集
#         mius:各类均值向量集合'''
#         M = datasets.shape[0]
#         C = []
#         for i in range(1):
#             dis = []
#             for j in range(self.k):
#                 print(self.p)
#                 dis.append(self.pdistCal(datasets[i,:], miuS[j,:]))
#                 print(dis)
#                 pos = np.where(dis == min(dis))
#                 c1 = pos[0]
#             C.append(c1)
#         return C
#     def cal(self, datasets):
#         '''初始化均值向量'''
#         A = np.random.randint(0, len(datasets), self.k)
#         # print(A)
#         u = datasets[A, :]
#         # print(u)
#         C = self.classCu(datasets, datasets[A, :])
#         return C

# #%%
# def pdistCal(x, y, p):
#     x = np.array(x).reshape(1,-1)
#     y = np.array(y)
#     '''计算距离'''
#     dis1 = np.sum((x - y) ** p) ** (1/p)
#     return dis1
# x = [1,2,3,4,5]
# u = np.array([[1,2,3,4,5],[2,3,4,5,6],[1,2,3,4,6],[2,3,4,5,5]])
# dis = []
# for i in range(u.shape[0]):
#     dis.append(pdistCal(x,u[i,:],p=1))
# dd = kmeans()
# C = dd.cal(X_train)
#%%
def pdistCal(x, y):
    x = np.array(x).reshape(1, -1)
    y = np.array(y).reshape(1, -1)
    '''计算距离'''
    dis1 = np.sum((x - y) ** p) ** (1 / p)
    return dis1

def MiuCal(x):
    '''计算均值向量'''
    x = np.array(x)
    u = np.mean(x, axis=1)
    return u

def classCu(datasets, miuS):
    '''计算均值向量
    datasets:数据集
    mius:各类均值向量集合'''
    M = datasets.shape[0]
    C = []
    for i in range(M):
        dis = []
        for j in range(k):
            dis.append(pdistCal(datasets[i, :], miuS[j, :]))
            pos = np.where(dis == min(dis))
            c1 = pos[0][0]
        C.append(c1)
    return np.array(C)


def changeMiu(datasets, C):
    datasets_X, datasets_Y = datasets.shape
    class1 = np.unique(C)
    u = np.zeros([len(class1), datasets_Y])
    for i in range(len(class1)):
        labels = class1[i]
        pos1 = np.where(C == labels)
        u[i,:] = MiuCal(datasets[pos1,:])
    return u

def cal(datasets):
    '''初始化均值向量'''
    A = np.random.randint(0, len(datasets), k)
    # print(A)
    u = datasets[A, :]
    # print(u)
    C = classCu(datasets, datasets[A, :])
    return np.array(C)

def train(datasets):
    A = np.random.randint(0, len(datasets), k)
    # A = np.array([8,34])
    # print(A)
    u0 = datasets[A, :]
    u1 = u0
    C = classCu(datasets, u0)
    u_change = changeMiu(datasets, C)

    while (u0 == u_change).all():
        u0 = u_change
        C = classCu(datasets, u0)
        u_change = changeMiu(datasets, C)
    return C, u1, u_change
def cal_accuracy(C, y_train):
    sim = C - y_train
    acc = np.sum(sim==0)/len(C)
    acc = max([acc, 1-acc])
    return acc
k = 2
p = 2

C, u1, u_re = train(X_train)

import matplotlib.pyplot as plt
# plt.plot(y_train)
# plt.plot(C)
acc = cal_accuracy(C, y_train)
print('聚类的准确率为：',acc)
#%%
# plt.plot(data[:,0],data[:,1],'*')
# 画出输入数据
plt.figure(1)
labels = y_train
data = X_train
plt.scatter(data[labels == 0,0], data[labels == 0,1], marker = 'o', color = 'k', s=100, label = 'c1')
plt.scatter(data[labels == 1,0], data[labels == 1,1], marker = 'o', color = 'g', s=100, label = 'c2')
# labels = to_categorical(labels, 2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
C = C.reshape(len(C))
plt.figure(2)
plt.scatter(data[C == 0,0], data[C == 0,1], marker = 'o', color = 'k', s=100, label = 'c1')
plt.scatter(data[C == 1,0], data[C == 1,1], marker = 'o', color = 'g', s=100, label = 'c2')
# labels = to_categorical(labels, 2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('DIY model')