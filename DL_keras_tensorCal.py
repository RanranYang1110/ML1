'''逐元素relu运算的简单实现'''
def naive_relu(x):
    c len(x.shape) == 2
    x = x.copy() #避免覆盖输入张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j], 0)
    return x
import numpy as np
data = np.array([[-1,2,3],[4,-5,6]])
y = naive_relu(data)
