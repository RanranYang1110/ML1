#-*- coding:utf-8 -*-
# @author: qianli
# @file: plit_norm.py
# @time: 2019/04/04
def normfun(x,mu,sigma):
    """
    计算正太分布的概率密度函数
    """
    pdf = np.exp( - ((x - mu) ** 2) / (2 * sigma **2 )) / (sigma * np.sqrt(2 * np.pi))
    return pdf

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    x = np.linspace(-5,5,5000)
    pdf = []
    mu = 0
    sigma = 2
    for i in x:
        y = normfun(i, mu, sigma)
        pdf.append(y)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(pdf, x, linewidth=3, color='k')
    x1 = 4 * np.ones([10,1])
    x2 = 3 * np.ones([10,1])
    x3 = 1 * np.ones([10, 1])
    x4 = -1 * np.ones([10, 1])
    t = np.linspace(0,0.25,10)
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(True)
    plt.plot(t,x1,'--r')

    plt.plot(t,x2,'--y')
    plt.plot(t,x3,'--b')
    plt.plot(t,x4,'--g')
    plt.show()
    plt.xlim([0,0.4])
    plt.ylim([-5,5])
