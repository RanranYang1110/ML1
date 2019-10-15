#-*- coding:utf-8 -*-
# @author: qianli
# @file: sklearn_bayes.py
# @time: 2019/03/28

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score,train_test_split
from sklearn.datasets import load_iris
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.2)
gnb = GaussianNB()
'''sklearn中共有3中不同类型的朴素贝叶斯
高斯分布型：用于classfication问题，假定属性/特征服从正态分布
伯努利型：最后得到的特征只有0（未出现）和1（出现过）
多项式型：用于离散值模型用，'''
scores = cross_val_score(gnb, X_train, y_train, cv=10)
print("accuracy:%.3f"%scores.mean())
#%%
#旧金山犯罪分类预测
import pandas as pd
import os
import numpy as np
os.chdir(r"D:\7-学习资料\a-study\datasets\Kaggle旧金山犯罪类型分类")
traindata = pd.read_csv('train.csv')
testdata = pd.read_csv('test.csv')
# 特征预处理
#%%
from sklearn.preprocessing import LabelEncoder
leCrime = LabelEncoder()
crime = leCrime.fit_transform(traindata.Category)
days = pd.get_dummies(traindata.DayOfWeek)
district = pd.get_dummies(traindata.PdDistrict)
hour = pd.to_datetime(traindata.Dates).dt.hour
hour = pd.get_dummies(hour)
trainData = pd.concat([hour, days, district], axis=1)
trainData['crime'] = crime
'''组合特征'''
days = pd.get_dummies(testdata.DayOfWeek)
district = pd.get_dummies(testdata.PdDistrict)
hour = pd.to_datetime(testdata.Dates).dt.hour
hour = pd.get_dummies(hour)
testData = pd.concat([hour, days, district], axis=1)
testData['crime'] = crime
#%% 训练模型
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import time
trainData.columns
features = trainData.columns[24:-1]
X_train, X_test, y_train, y_test = train_test_split(trainData[features], trainData.crime,
                                                    test_size=0.3)
NB = BernoulliNB()
nbStart = time.time()
NB.fit(X_train, y_train)
nbCostTime = time.time() - nbStart
propa = NB.predict_proba(X_test)

print("朴素贝叶斯建模%.2f秒"%(nbCostTime))
predicted = np.array(propa)
logLoss=log_loss(y_test, predicted)
print("朴素贝叶斯的log损失为:%.6f"%logLoss)
#%% sklearn bayes
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
import os
os.chdir(r"D:\7-学习资料\a-study\python\python\00  Python机器学习经典实例代码\Chapter02")
def plot_classifier(classifier, X, y):
    # define ranges to plot the figure
    x_min, x_max = min(X[:, 0]) - 1.0, max(X[:, 0]) + 1.0
    y_min, y_max = min(X[:, 1]) - 1.0, max(X[:, 1]) + 1.0

    # denotes the step size that will be used in the mesh grid
    step_size = 0.01

    # define the mesh grid
    x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size), np.arange(y_min, y_max, step_size))

    # compute the classifier output
    mesh_output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

    # reshape the array
    mesh_output = mesh_output.reshape(x_values.shape)

    # Plot the output using a colored plot
    plt.figure()

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks((np.arange(int(min(X[:, 0])-1), int(max(X[:, 0])+1), 1.0)))
    plt.yticks((np.arange(int(min(X[:, 1])-1), int(max(X[:, 1])+1), 1.0)))

    plt.show()

input_file = 'data_multivar.txt'
X = []
y = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = [float(x) for x in line.split(',')]
        X.append(data[:-1])
        y.append(data[-1])
X = np.array(X)
y = np.array(y)
classifier_gaussiannb = GaussianNB()
classifier_gaussiannb.fit(X, y)
y_pred = classifier_gaussiannb.predict(X)
accuracy = 100.0 * (y == y_pred).sum() / X.shape[0]
print('accuracy of the classifier=', round(accuracy,2),'%')
plot_classifier(classifier_gaussiannb,X,y)

