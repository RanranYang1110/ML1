#-*- coding:utf-8 -*-
# @author: qianli
# @file: Recurrent_network.py
# @time: 2019/03/27
import numpy as np
import matplotlib.pyplot as plt
import neurolab as nl

#定义一个函数，该函数利用输入参数创建一个波形
def create_waveform(num_points):
    #创建训练样本
    data1 = 1 * np.cos(np.arange(0, num_points))
    data2 = 2 * np.cos(np.arange(0, num_points))
    data3 = 3 * np.cos(np.arange(0, num_points))
    data4 = 4 * np.cos(np.arange(0, num_points))
    # 创建不同的振幅
    amp1 = np.ones(num_points)
    amp2 = 4 + np.zeros(num_points)
    amp3 = 2 * np.ones(num_points)
    amp4 = 0.5 + np.zeros(num_points)
    data = np.array([data1, data2, data3, data4]).reshape(num_points * 4, 1)
    amplitude = np.array([[amp1, amp2, amp3, amp4]]).reshape(num_points * 4, 1)
    return data, amplitude

# 使用网络画出输出结果
def draw_output(net, num_points_test):
    data_test, amplitude_test = create_waveform(num_points_test)
    output_test = net.sim(data_test)
    plt.plot(amplitude_test.reshape(num_points_test * 4))
    plt.plot(output_test.reshape(num_points_test * 4))

if __name__=='__main__' :
    # 获取数据
    num_points = 30
    data, amplitude = create_waveform(num_points)
    # 创建一个两层的神经网络
    net = nl.net.newelm([[-2, 2]], [10, 1], [nl.trans.TanSig(), nl.trans.PureLin()])
    # 设定初始化函数并进行初始化
    net.layers[0].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    net.layers[1].initf = nl.init.InitRand([-0.1, 0.1], 'wb')
    net.init()
    # 训练递归神经网络
    error = net.train(data, amplitude, epochs=1000, show=100, goal=0.01)
    # 计算来自网络的输出
    output = net.sim(data)
    # 画出训练结果
    plt.subplot(211)
    plt.plot(error)
    plt.xlabel('Number of epochs')
    plt.ylabel('Error (MSE)')
    plt.subplot(212)
    plt.plot(amplitude.reshape(num_points * 4))
    plt.plot(output.reshape(num_points * 4))
    plt.legend(['Ground truth', 'Predicted output'])
    # 在多个尺度上对未知数据进行测试
    plt.figure()
    plt.subplot(211)
    draw_output(net, 74)
    plt.xlim([0, 300])
    plt.subplot(212)
    draw_output(net, 54)
    plt.xlim([0, 300])
    plt.show()