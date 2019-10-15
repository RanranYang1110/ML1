#-*- coding:utf-8 -*-
# @author: qianli
# @file: LR.py
# @time: 2019/03/21
#%%
import numpy as np
class logistic(object):
    def __init__(self):
        self.W = None
    def train(self, X, y, learn_rate = 0.01, num_iters = 5000):
        num_train, num_features = X.shape
        self.W = 0.001 * np.random.randn(num_features,1).reshape((-1,1))
        loss = []
        for i in range(num_iters):
            error, dw = self.compute_loss(X, y)
            self.W += -learn_rate * dw
            loss.append(error)
            if i%200 == 0:
                print('i=%d, error = %f'%(i,error))
        return loss
    def compute_loss(self, X, y):
        '''
        calculate the cost function
        :param X: the input array
        :param y: the label
        :return: loss, dw
        '''
        num_train = X.shape[0]
        h = self.output(X)
        loss = - np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        loss = loss / num_train
        dw = X.T.dot((h - y)) / num_train
        return loss, dw

    def output(self, X):
        g = np.dot(X, self.W)
        return self.sigmoid(g)

    def sigmoid(self, X):
        return 1. / (1 + np.exp(-X))

    def predict(self, X_test):
        h = self.output(X_test)
        y_pred = np.where(h >= 0.5 , 1, 0)
        return y_pred

import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

iris = load_iris()
data = iris.data
target = iris.target
X = data[0:100, [0,2]]
y = target[0:100]
print(X[:5])
print(y[-5:])
label = np.array(y)
index_0 = np.where(label == 0)
plt.scatter(X[index_0, 0], X[index_0,1], marker='x', color='b', label=0, s=15)
index_1 = np.where(label == 1)
plt.scatter(X[index_1, 0], X[index_1,1], marker='o', color='r', label=1, s=15)
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc = 'upper left')
plt.show()
#%%
y = y.reshape(-1,1)
one = np.ones((X.shape[0],1))
X_train = np.hstack((one, X))
classify = logistic()
loss = classify.train(X_train,y)
print(classify.W)
plt.figure(2)
plt.plot(loss)
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()
##