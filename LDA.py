#-*- coding:utf-8 -*-
# @author: qianli
# @file: LDA.py
# @time: 2019/03/22
import numpy as np
def LDA(X, y):
    x1 = X[np.where(y == 1)[0],:]
    x2 = X[np.where(y == 0)[0],:]
    u1 = np.mean(x1, axis=0)
    u2 = np.mean(x2, axis=0)
    conv1 = np.dot((x1 - u1).T,(x1 - u1)) #计算第一类的类内散度矩阵
    conv2 = np.dot((x2 - u2).T,(x2 - u2)) #计算第二类的类内散度矩阵
    Sw = conv1 + conv2
    w = np.dot(np.mat(Sw).I, (u1 - u2).reshape(len(u1),1)) #求w
    x1_new = func(x1, w)
    x2_new = func(x2, w)
    y1_new = np.ones(x1.shape[0])
    y2_new = np.zeros(x2.shape[0])
    return x1_new, x2_new, y1_new, y2_new

def func(x, w):
    return np.dot((x), w)

from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data[:100,[0,2]]
y = iris.target[:100]
x1_new, x2_new, y1_new, y2_new = LDA(X,y)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
plt.show()
plt.plot(x1_new, y1_new, 'b*')
plt.plot(x2_new, y2_new, 'ro')
plt.show()
