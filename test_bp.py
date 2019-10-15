#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_bp.py
# @time: 2019/03/26

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
iris = load_iris()
data = iris.data
target = iris.target
plt.plot(target)
# X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.33)
learingRate = 0.05
#%%
from sklearn.datasets import load_digits
from sklearn.preprocessing import LabelBinarizer #标签二值化
import numpy as np
import pylab as pl
def sigmoid(x): #激活函数sigmoid
    return 1./(1. + np.exp(-x))
def dsigmoid(x): #sigmoid的倒数
    return x * (1-x)
class NeuralNetwork:
    def __init__(self,layers): #这里是三层网络，列表[64,100,10]表示输入，隐藏，输出层的单元个数
        #初始化权值，范围为1~-1
        self.v = np.random.random((layers[0]+1, layers[1])) * 2 - 1
         #隐藏层权值，包含64*100个权值，100个隐层神经元的阈值
        self.w = np.random.random((layers[1], layers[2])) * 2 - 1
    def train(self, X, y, lr=0.1, epochs=20000):
        #为数据集添加偏置
        temp = np.ones([X.shape[0],X.shape[1]+1])
        temp[:,:-1] = X
        X = temp

        #进行权值训练更新
        for i in range(epochs + 1):
            m = np.random.randint(X.shape[0]) #随机选取一行数据进行更新
            x = X[m].reshape(1,-1)
            # x = np.atleast_2d(x) #转为二维数据
            L1 = sigmoid(np.dot(x, self.v)) #隐层输出（1，100）
            L2 = sigmoid(np.dot(L1, self.w)) #输出层输出（1，10）

            #delta
            L2_delta = (y[m]-L2) * dsigmoid(L2) #(1,10)
            L1_delta = L2_delta.dot(self.w.T) * dsigmoid(L1) #(1,100),这里是数组的乘法，对应元素相乘

            #更新
            self.w += lr * L1.T.dot(L2_delta)  # (100,10)
            self.v += lr * x.T.dot(L1_delta)  #

            # 每训练1000次预测准确率
            if i % 1000==0:
                predictions = []
                for j in range(X_test.shape[0]):
                    out = self.predict(X_test[j])  # 用验证集去测试
                    predictions.append(np.argmax(out))  # 返回预测结果
                accuracy = np.mean(np.equal(predictions, y_test))  # 求平均值
                print('epoch:', i, 'accuracy:', accuracy)
    def predict(self, x):
        #添加转置，这里是一维的
        temp = np.ones(x.shape[0] + 1)
        temp[:-1] = x
        x = temp
        x = np.atleast_2d(x)
        L1 = sigmoid(np.dot(x, self.v))  # 隐层输出
        L2 = sigmoid(np.dot(L1, self.w))  # 输出层输出
        return L2
digits = load_digits()#载入数据
X = digits.data#数据
y = digits.target#标签
# #print y[0:10]
# #数据归一化,一般是x=(x-x.min)/x.max-x.min
X -= X.min()
X /= X.max()
#创建神经网络
nm = NeuralNetwork([64,100,10])
X_train, X_test, y_train, y_test=train_test_split(X,y)#默认分割：3:1
#标签二值化
labels_train = LabelBinarizer().fit_transform(y_train)
print(labels_train[0:10])
labels_test = LabelBinarizer().fit_transform(y_test)
print('start')
nm.train(X_train,labels_train,epochs=20000)
print('end')
