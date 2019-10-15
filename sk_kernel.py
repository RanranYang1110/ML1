#-*- coding:utf-8 -*-
# @author: qianli
# @file: sk_kernel.py
# @time: 2019/06/18
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
         * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
for hyperparameter in kernel.hyperparameters: print(hyperparameter)
params = kernel.get_params()
for key in sorted(params):
    print("%s : %s" % (key, params[key]))
    # print(key)
print(kernel.theta)
print(kernel.bounds)
# ConstantKernel内核类可以被用作Product内核类的一部分，在它可以对其他因子进行
#度量的场景下或者作为更改高斯过程均值的

#whiteKernel内核类的主要应用实例在于当解释信号的噪声部分时，可以作为内核集合的一部分
#通过调节参数noise_level,该类可以用来估计噪声级别

#径向基函数内核 通过定长的参数l>0对内核进行参数话
#Matern内核 控制结果函数的平滑程度
#RationalQuadratic 不同特征尺度下的RBF内核的规模混合
#ExpSineSquared（正弦平方内核） 对周期性函数进行建模
#DotProduct内核通常和指数相结合

'''回归实例介绍'''
import numpy as np
from sklearn import gaussian_process
def f(x):
    return x * np.sin(x)
X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(0, 10, 1000)).T
gp = gaussian_process.GaussianProcess(theta0=1e-2, thetaL=1e-4, thetaU=1e-1)
gp.fit(X, y)

