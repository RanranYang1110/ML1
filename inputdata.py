#-*- coding:utf-8 -*-
# @author: qianli
# @file: inputdata.py
# @time: 2019/04/09

import numpy as np
import os
import scipy.io as sio

def inputdata(data, Expected_len):
    '''对数据进行插值
    input: data-需进行处理的数据，格式为array型
           Expected_len - 预期的长度
    output: dd - 填充之后的数据
    '''
    dd = np.zeros([Expected_len,1])
    j = 0
    N = len(data)
    for i in range(0,N-1):
        mm = int(i * Expected_len / N)
#        print(i)
        dd[mm,0] = data[i][0]
    pos = np.where(dd != 0)[0]
    for j in range(len(pos)-1):
        index1 = pos[j]
        index2 = pos[j+1]
        num = index2 - index1
        for k in range(1,num):
            dd[index1+k,0] = (dd[index1+k-1,0] + dd[index2,0])/2
    dd = dd.reshape(Expected_len)
    return dd
#%%
filename=r'C:\Users\qianli\Desktop\dd1.mat'
sj = sio.loadmat(filename)
data1 = sj['data1'][:970]
filename2 = r'C:\Users\qianli\Desktop\dd2.mat'
sj = sio.loadmat(filename2)
data2 = sj['data2']
data3 = inputdata(data2, len(data1))
t1 = np.linspace(0,len(data3),len(data3)-5)/(len(data3)-5)
y1 = np.ones([len(data3)-5,1])
y2 = 0.4 * y1
y3 = 1.2 * y1
#%%
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(1, 1, 1)
f1 = plt.plot(t1,data1[:-5])
f2 = plt.plot(t1,data3[:-5])
plt.plot(t1, y1 ,'--r')
plt.plot(t1, y2, '--g')
plt.plot(t1, y3,'--y')

plt.xlim([0, 1])
# plt.xticks([])
# plt.yticks([])
# ax.spines['top'].set_visible(False)
# ax.spines['right'].set_visible(False)
# ax.spines['bottom'].set_visible(False)
plt.legend(['test1','test2'])