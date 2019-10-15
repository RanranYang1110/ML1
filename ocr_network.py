#-*- coding:utf-8 -*-
# @author: qianli
# @file: ocr_network.py
# @time: 2019/03/27
'''通过神经网络创建一个光学字符识别器'''
'''在光学字符识别数据库中将字符可视化，并建立一个识别器'''
import numpy as np
import neurolab as nl
import os
import sys
import cv2
os.chdir(r"D:\7-学习资料\a-study\python\python\00  Python机器学习经典实例代码\Chapter11\letter.data")
input_file = 'letter.data'
#定义可视化参数
scaling_factor = 10
start_index = 6
end_index = -1
h,w = 16,8
with open(input_file, 'r') as f:
    for line in f.readlines():
        data = np.array([255 * float(x) for x in line.split('\t')[start_index:end_index]])
        # 数组调整为所需的形状
        img = np.reshape(data, (h,w))
        img_scaled = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor)
        cv2.imshow('Image', img_scaled)
        c = cv2.waitKey()
        if c == 27:
            break