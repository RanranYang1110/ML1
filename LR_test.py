#-*- coding:utf-8 -*-
# @author: qianli
# @file: LR_test.py
# @time: 2019/03/21
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

class LossRegression(object):
    def __init__(self):
        self.W = None
    def output(self,X):
        '''
        将输入矩阵转化为theta^TX,并经过sigmoid函数
        :param X: 数据矩阵
        :return: sigmoid之后的值
        '''
        theta = np.dot(X,self.W)
        return self.sigmoid(theta)
    def sigmoid(self,T):
        return 1./(1 + np.exp(-T))

    def calculate_coss(self,X,y):
        m = X.shape[0]
        y1 = self.output(X)
        coss = - np.sum(y *np.log(y1) + (1-y)*np.log(1-y1)) / m
        dw = X.T.dot((y1 - y)) / m
        return coss, dw
    def train(self, X, y, learing_rate=0.01, epochs=5000):
        loss = []
        m, n = X.shape
        self.W = 0.001 * np.random.randn(n,1).reshape((-1,1))
        for i in range(epochs):
            coss, dw = self.calculate_coss(X, y)
            self.W += - learing_rate * dw
            loss.append(coss)
        return loss
    def predict(self, X_test):
        y1 = self.output(self,X_test)
        y_pred = np.where(y1 >= 0.5, 1, 0)
        # if y1 > 0.5:
        #     y_pred = 1
        # else:
        #     y_pred = 0
        return y_pred

if __name__ == '__main__':
    iris = load_iris()
    data = iris.data
    target = iris.target
    X = data[0:100, [0, 2]]
    y = target[0:100]
    print(X[:5])
    print(y[-5:])
    label = np.array(y)
    index_0 = np.where(label == 0)
    plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', color='b', label=0, s=15)
    index_1 = np.where(label == 1)
    plt.scatter(X[index_1, 0], X[index_1, 1], marker='o', color='r', label=1, s=15)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend(loc='upper left')
    plt.show()

    y = y.reshape(-1, 1)
    one = np.ones((X.shape[0], 1))
    X_train = np.hstack((one, X))
    classify = LossRegression()
    loss = classify.train(X_train, y)
    print(classify.W)
    plt.figure(2)
    plt.plot(loss)
    '''可视化决策边界'''
    label = np.array(y)
    index_0 = np.where(label == 0)
    plt.scatter(X[index_0, 0], X[index_0, 1], marker='x', color='b', label=0, s=15)
    index_1 = np.where(label == 1)
    plt.scatter(X[index_1, 0], X[index_1, 1], marker='o ', color='r', label=1, s=15)
    x1 = np.arange(4, 7.5, 0.5)
    x2 = (- classify.W[0] - classify.W[1]*x1) / classify.W[2]
    plt.plot(x1, x2, color='black')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()