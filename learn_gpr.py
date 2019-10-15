#-*- coding:utf-8 -*-
# @author: qianli
# @file: learn_gpr.py
# @time: 2019/07/10

'''kernel主要是用来计算数据点之间的高斯过程协方差
内核通过超参数向量进行参数化，超参数可以控制例如内核的长度或周期性
通过设置__call__方法的参数eval_gradient=True, 所有的内核支持计算解析内核自协方差
对于超参数的解析梯度'''

#基础内核
# constantKernel内核类可以被用作Product内核类的一部分，在它可以对其他因子（内核）
#进行度量的场景下或者作为更改高斯过程均值的Sum类的一部分。这取决于参数constant_value的设置。

from sklearn.gaussian_process.kernels import ConstantKernel, RBF
kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) \
                         * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) \
                         + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
for hyperparameter in kernel.hyperparameters: print(hyperparameter)
params = kernel.get_params()
for key in sorted(params): print("%s : %s" % (key, params[key]))
print(kernel.theta)
print(kernel.bounds)
#%%
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
X, y = make_friedman2(n_samples=500, noise=0, random_state=0)
kernel = DotProduct() + WhiteKernel()
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)
gpr.score(X, y)
gpr.predict(X[:2,:],return_std=True)
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
np.random.seed(1)
def f(x):
    return x * np.sin(x)

X = np.atleast_2d([1., 3., 5., 6., 7., 8.]).T
y = f(X).ravel()

x = np.atleast_2d(np.linspace(0, 10, 1000)).T
kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(X, y)
y_pred, sigma = gp.predict(x, return_std=True)
plt.figure()
plt.plot(x, f(x), 'r:', label=r'$f(x)=x\,sin(x)$')
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(x, y_pred, 'b-', label='Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.96 * sigma,
                         (y_pred + 1.96 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')
#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, RationalQuadratic, ExpSineSquared

def load_mauna_loa_atmospheric_co2():
    ml_data = fetch_openml(data_id=41187)
    months = []
    ppmv_sums = []
    counts = []
    y = ml_data.data[:, 0]
    m = ml_data.data[:, 1]
    month_float = y + (m - 1) / 12
    ppmvs = ml_data.target

    for month, ppmv in zip(month_float, ppmvs):
        if not months or month != months[-1]:
            months.append(month)
            ppmv_sums.append(ppmv)
            counts.append(1)
        else:
            ppmv_sums[-1] += ppmv
            counts[-1] += 1
    months = np.asarray(months).reshape(-1,1)
    avg_ppmvs = np.asarray(ppmv_sums)/counts
    return months, avg_ppmvs
X, y = load_mauna_loa_atmospheric_co2()
k1 = 66.0 ** 2 * RBF(length_scale=67.0)
k2 = 2.4 ** 2 * RBF(length_scale=90.0) * ExpSineSquared(length_scale=1.3, periodicity=1.0)
k3 = 0.66 **2 * RationalQuadratic(length_scale=1.2, alpha=0.78)
k4 = 0.18 ** 2 * RBF(length_scale=0.134) + WhiteKernel(noise_level=0.19 ** 2)
kernel_gpml = k1 + k2 + k3 + k4
gp = GaussianProcessRegressor(kernel=kernel_gpml, alpha=0, optimizer=None, normalize_y=True)
gp.fit(X, y)
print('GPML kernel: %s' % gp.kernel_)
print('Log-marginal-likeliood: %.3f' % gp.log_marginal_likelihood(gp.kernel_.theta))
k1 = 50.0 ** 2 * RBF(length_scale=50.0)
k2 = 2.0 ** 2 * RBF(length_scale=100.0) * ExpSineSquared(length_scale=1.0, periodicity=1.0, periodicity_bounds='fixed')
k3 = 0.5 ** 2 * RationalQuadratic(length_scale=1.0, alpha=1.0)
k4 = 0.1 ** 2 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1**2, noise_level_bounds=(1e-3, np.inf))
kernel = k1 + k2 + k3 + k4
gp = GaussianProcessRegressor(kernel=kernel, alpha=0, normalize_y=True)
gp.fit(X, y)
print('\nLearned kernel: %s' % gp.kernel_)
print('Log-marginal-likelihood: %.3f' % gp.log_marginal_likelihood(gp.kernel_.theta))
X_ = np.linspace(X.min(), X.max() + 30, 1000)[:, np.newaxis]
y_pred, y_std = gp.predict(X_, return_std=True)

plt.scatter(X, y, c='k')
plt.plot(X_, y_pred)
plt.fill_between(X_[:, 0], y_pred - y_std, y_pred + y_std, alpha=0.5, color='k')
plt.xlim(X_.min(), X_.max())
plt.xlabel("Year")
plt.ylabel(r"CO$_2$ in ppm")
plt.title(r"Atmospheric CO$_2$ concentration at Mauna Loa")
plt.tight_layout()
plt.show()

#%%
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, ExpSineSquared

rng = np.random.RandomState(0)
X = 15 * rng.rand(100, 1)
y = np.sin(X).ravel()
y += 3 * (0.5 - rng.rand(X.shape[0])) # add noise

param_grid = {'alpha': [1e0, 1e-1, 1e-2, 1e-3],
              'kernel':[ExpSineSquared(1, p)
                        for l in np.logspace(-2, 2, 10)
                        for p in np.logspace(0, 2, 10)]}
kr = GridSearchCV(KernelRidge(), cv=5, param_grid=param_grid)
stime = time.time()
kr.fit(X, y)
print('Time for GPR fitting: %.3f' % (time.time() - stime))
gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(1e-1)
gpr = GaussianProcessRegressor(kernel=gp_kernel)
stime = time.time()
gpr.fit(X, y)
print('Time for GPR fitting: %.3f' % (time.time() - stime))

X_plot = np.linspace(0, 20, 1000)[:, None]
stime = time.time()
y_kr = kr.predict(X_plot)
print('Time for KRR prediction: %.3f' % (time.time() - stime))

stime = time.time()
y_gpr = gpr.predict(X_plot, return_std=False)
print('Time for GPR prediction: %.3f' % (time.time() - stime))

stime = time.time()
y_gpr, y_std = gpr.predict(X_plot, return_std=True)
print('Time for GPR prediction: %.3f' % (time.time() - stime))

plt.figure(figsize=(10, 5))
lw = 2
plt.scatter(X, y, c='k', label='data')
plt.plot(X_plot, np.sin(X_plot), color='navy', lw = lw, label='True')
plt.plot(X_plot, y_kr, color='turquoise', lw=lw, label='KRR(%s)' % kr.best_params_)
plt.plot(X_plot, y_gpr, color='darkorange', lw=lw, label='GPR(%s)' % kr.best_params_)
plt.fill_between(X_plot[:, 0], y_gpr - y_std, y_gpr + y_std, color='darkorange', alpha=0.2)
plt.xlabel('data')
plt.ylabel('target')
plt.xlim(0, 20)
plt.ylim(-4, 4)
plt.title('GPR versus Kernel Ridge')
plt.legend(loc="best",  scatterpoints=1, prop={'size': 8})
plt.show()

import pandas as pd
import numpy as np
data = np.array([[1,2,3],[4,5,6]])
df = pd.DataFrame(data, columns=['x','y','z'])