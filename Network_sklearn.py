#-*- coding:utf-8 -*-
# @author: qianli
# @file: Network_sklearn.py
# @time: 2019/03/26

import numpy as np
import neurolab as nl
import matplotlib.pyplot as plt

# 创建一个单层神经网络
from sklearn.datasets import load_iris
from keras.utils import to_categorical

# 导入数据
iris = load_iris()
data = iris.data[:100,:2]
labels = iris.target[:100]

# 画出输入数据
plt.figure()
plt.scatter(data[labels == 0,0], data[labels == 0,1], marker = 'o', color = 'k', s=100, label = 'c1')
plt.scatter(data[labels == 1,0], data[labels == 1,1], marker = 'o', color = 'g', s=100, label = 'c2')
labels = to_categorical(labels, 2)
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Input data')
#提取每个维度的最小值和最大值
x_min, x_max = data[:,0].min(), data[:,0].max()
y_min, y_max = data[:,1].min(), data[:,1].max()

#定义一个单层神经网络，包含两个神经元
single_layer_net = nl.net.newp([[x_min, x_max],[y_min, y_max]],2)
#通过50次迭代训练神经网络
error = single_layer_net.train(data, labels, epochs=50, show=15, lr=0.01)

# 画出结果
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Training error')
plt.title('Training error progress')
plt.grid()
plt.show()

#用新的测试数据来测试神经网络：
print(single_layer_net.sim([[5,3.3]]))
#%%
'''创建一个深度神经网络'''
# 生成训练数据
min_value = -12
max_value = 12
num_datapoints = 90
x = np.linspace(min_value, max_value, num_datapoints)
y = 2 * np.square(x) + 7
y /= np.linalg.norm(y)
data = x.reshape(num_datapoints, 1)
labels = y.reshape(num_datapoints, 1)
# 画出输入数据
# plt.figure()
# plt.scatter(data, labels)
# plt.xlabel('X-axis')
# plt.ylabel('Y-axis')
# plt.title('Input data')
# 定义一个深度神经网络，该神经网络包含两个隐藏层，每个隐藏层包含10个神经元
# 输出层由一个神经元组成
multilayer_net = nl.net.newff([[min_value, max_value]], [10, 10, 1])
# 设置训练算法为梯度下降法
multilayer_net.trainf = nl.train.train_gd
error = multilayer_net.train(data, labels, epochs=2000, show=1000, goal=0.01)
# 用训练数据运行该网络，预测结果
predicted_output = multilayer_net.sim(data)
# 画出训练误差结果
plt.figure()
plt.plot(error)
plt.xlabel('Number of epochs')
plt.ylabel('Error')
plt.title('Training error progress')
# 画出预测结果
x2 = np.linspace(min_value, max_value, num_datapoints * 2)
y2 = multilayer_net.sim(x2.reshape(x2.size,1)).reshape(x2.size)
y3 = predicted_output.reshape(num_datapoints)
plt.figure()
plt.plot(x2, y2, '-', x, y, '.', x, y3, 'p')
#%%
'''创建一个向量量化器'''
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl
import os
os.chdir(r"D:\7-学习资料\a-study\python\python\00  Python机器学习经典实例代码\Chapter11")
input_file = "data_vq.txt"
input_text = np.loadtxt(input_file)
data = input_text[:, :2]
labels = input_text[:, 2:]
#定义一个两层的学习向量量化(learning vector quantization,LVQ)神经网络。
#函数中最后一个参数的数组指定了每个输出的加权百分比（各加权百分比之和应为1）
net = nl.net.newlvq(nl.tool.minmax(data), 10, [0.25, 0.25, 0.25, 0.25])
error = net.train(data, labels, epochs=100, goal=-1)
#创建输入网格
xx, yy = np.meshgrid(np.arrange(0,8,0.2), np.arrange(0,8,0.2))
xx.shape = xx.size, 1
yy.shape = yy.size, 1
input_grid = np.concatenate((xx, yy), axis=1)
#用这些网格点值评价网络
output_grid = net.sim(input_grid)
# 在数据中定义四个类:
class1 = data[labels[:,0] == 1]
class2 = data[labels[:,1] == 1]
class3 = data[labels[:,2] == 1]
class4 = data[labels[:,3] == 1]
#为四个类定义网格：
grid1 = input_grid[output_grid[:,0] == 1]
grid2 = input_grid[output_grid[:,1] == 1]
grid3 = input_grid[output_grid[:,2] == 1]
grid4 = input_grid[output_grid[:,3] == 1]
# 画出输出结果
plt.plot(class1[:,0], class1[:,1], 'ko', class2[:,0], class2[:,1], 'ko',
class3[:,0], class3[:,1], 'ko', class4[:,0], class4[:,1], 'ko')
plt.plot(grid1[:,0], grid1[:,1], 'b.', grid2[:,0], grid2[:,1], 'gx',
grid3[:,0], grid3[:,1], 'cs', grid4[:,0], grid4[:,1], 'ro')
plt.axis([0, 8, 0, 8])
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Vector quantization using neural networks')
plt.show()