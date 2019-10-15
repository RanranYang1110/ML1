#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_pca.py
# @time: 2019/06/11

import numpy as np
import scipy.io as sio
path = r"D:\1-轨交行业\5-中车永济轴承PHM\3-培训材料准备\2-特征工程\特征融合PCA\testdata1.mat"
data = sio.loadmat(path)
data = data['data']
'''去平均值'''
meanData = np.mean(data, axis=0)
data1 = data - meanData
'''计算协方差矩阵'''
covMat = np.cov(data1, rowvar=0)
#%%
'''计算特征向量和特征值'''
eigVals, eigVects = np.linalg.eig(np.mat(covMat))
eigValInd = np.argsort(eigVals)
eigValInd = eigValInd[::-1]
redEigVects = eigVects[:,eigValInd]
'''将数据转换到新的空间'''
lowDDataMat = data1 * redEigVects
reconMat = lowDDataMat * redEigVects.T + meanData

#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(data)
newX = pca.fit_transform(data) #降维后的数据
# print(newX)
# print(reconMat)
import matplotlib.pyplot as plt
plt.plot(newX[:,0])
plt.plot(reconMat[:,0])
plt.legend(['sk','diy'])