#-*- coding:utf-8 -*-
# @author: qianli
# @file: svm.py
# @time: 2019/03/27


import os
os.chdir(r"D:\7-学习资料\a-study\python\python\00  Python机器学习经典实例代码\Chapter03")
input_file = 'data_multivar.txt'
import sys
sys.path.append(r"D:\7-学习资料\a-study\python\python\00  Python机器学习经典实例代码\Chapter11")
# from utilities import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# Load multivar data in the input file
def load_data(input_file):
    X = []
    y = []
    with open(input_file, 'r') as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])

    X = np.array(X)
    y = np.array(y)

    return X, y

# Plot the classifier boundaries on input data
def plot_classifier(classifier, X, y, title='Classifier boundaries', annotate=False):
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

    # Set the title
    plt.title(title)

    # choose a color scheme you can find all the options
    # here: http://matplotlib.org/examples/color/colormaps_reference.html
    plt.pcolormesh(x_values, y_values, mesh_output, cmap=plt.cm.gray)

    # Overlay the training points on the plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

    # specify the boundaries of the figure
    plt.xlim(x_values.min(), x_values.max())
    plt.ylim(y_values.min(), y_values.max())

    # specify the ticks on the X and Y axes
    plt.xticks(())
    plt.yticks(())

    if annotate:
        for x, y in zip(X[:, 0], X[:, 1]):
            # Full documentation of the function available here:
            # http://matplotlib.org/api/text_api.html#matplotlib.text.Annotation
            plt.annotate(
                '(' + str(round(x, 1)) + ',' + str(round(y, 1)) + ')',
                xy = (x, y), xytext = (-15, 15),
                textcoords = 'offset points',
                horizontalalignment = 'right',
                verticalalignment = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.6', fc = 'white', alpha = 0.8),
                arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=0'))

# Print performance metrics
def print_accuracy_report(classifier, X, y, num_validations=5):
    accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=num_validations)
    print("Accuracy: " + str(round(100*accuracy.mean(), 2)) + "%")

    f1 = cross_val_score(classifier, X, y, scoring='f1_weighted', cv=num_validations)
    print ("F1: " + str(round(100*f1.mean(), 2)) + "%")

    precision = cross_val_score(classifier, X, y, scoring='precision_weighted', cv=num_validations)
    print ("Precision: " + str(round(100*precision.mean(), 2)) + "%")

    recall = cross_val_score(classifier, X, y, scoring='recall_weighted', cv=num_validations)
    print ("Recall: " + str(round(100*recall.mean(), 2)) + "%")

def input_data(input_file):
    X = []
    y = []
    with open(input_file) as f:
        for line in f.readlines():
            data = [float(x) for x in line.split(',')]
            X.append(data[:-1])
            y.append(data[-1])
    X = np.array(X)
    y = np.array(y)
    return X,y
X,y = input_data(input_file)
#画图
# plt.figure()
# plt.scatter(X[y == 0,0], X[y == 0,1], marker = 'o', color = 'r', s=100, label = 'c1')
# plt.scatter(X[y == 1,0], X[y == 1,1], marker = 'o', color = 'g', s=100, label = 'c2')
# # labels = to_categorical(y, 2)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Input data')

'''导入模型'''
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate

from sklearn.svm import SVC
from sklearn.svm import LinearSVC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
#使用线性和函数初始化一个SVM模型
params = {'kernel': 'linear'} #使用线性核函数，建立线性分类器
params = {'kernel': 'rbf'} #建立高斯核函数
classifier = SVC(**params)
# classifier = LinearSVC()
classifier.fit(X_train,y_train)
#训练集结果
plot_classifier(classifier, X_train, y_train, 'Training dataset')
plt.show()
y_test_pred = classifier.predict(X_test)
#测试集结果
plot_classifier(classifier, X_test, y_test, 'Test dataset')
plt.show()
#计算训练结果的准确性
from sklearn.metrics import classification_report
target_names = ['Class-' + str(int(i)) for i in set(y)]
print("\n" + "#"*30)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(X_train),target_names=target_names))
print ("#"*30 + "\n")
#在测试集上的准确性
print("#"*30)
print("\nClassification report on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=target_names))
print("#"*30 + "\n")

#%%
from sklearn import svm
X = [[0, 0],[1, 1]]
y = [0,1]
clf = svm.SVC()
clf.fit(X,y)
clf.predict([[2. , 2.]])
# 获得支持向量
clf.support_vectors_
# 获得支持向量的索引get indeices of suppport vectors
clf.support_
# 为每一个类别获得支持向量的数量
clf.n_support_
'''多元分类'''
X = [[0],[1],[2],[3]]
Y = [0, 1, 2, 3]
clf = svm.SVC(decision_function_shape='ovo') #1对1分类，需要4*3/2个分类器
clf = svm.SVC(decision_function_shape='ovr') #1对其余分类，需要4个分类器
clf.fit(X, Y)
dec = clf.decision_function([[1]])
dec.shape[1]

'''LinearSVC实现ovr的多类别策略'''
X = [[0, 0],[1, 1]]
y = [0,1]
lin_clf = svm.LinearSVC()
lin_clf.fit(X, y)
dec = lin_clf.decision_function([[2,2]])
# dec.shape[1]
#%%
from sklearn import svm
# model = svm.SVC(kernel='rbf', C=1, gamma=1)
gamma = 5
model = svm.SVC(kernel='rbf', C=1, gamma=gamma)
model.fit(X_train, y_train)
model.score(X_test, y_test)
plot_classifier(model, X_test, y_test, 'gamma:'+str(gamma))