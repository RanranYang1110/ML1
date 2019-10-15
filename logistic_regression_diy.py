#-*- coding:utf-8 -*-
# @author: qianli
# @file: logistic_regression_diy.py
# @time: 2019/03/20
#%%
import numpy as np
def likelihood_sub(x, y, beta):
    '''
    @param x: one sample variables
    @param y: one sample labels
    @param beta: the parameter vector in 3.27
    @return: the sub_log-likelihood of 3.27
    '''
    return -y * np.dot(beta, x.T) + np.math.log(1+np.math.exp(np.dot(beta, x.T)))

def likelihood(X, y, beta):
    '''
    @:param X: the sample variables matrix
    @:param y: the sample label matrix
    @:param beta: the parameter vector in 3.27
    @:return: the log-likelihood of 3.27
    '''
    sum = 0
    m, n = np.shape(X)
    for i in range(m):
        sum += likelihood(X[i],y[i], beta)
    return sum

def partial_derivative(X, y, beta): #refer to 3.30 on book page 60
    '''
    :param X: the sample variables matrix
    :param y: the sample label matrix
    :param beta: the parameter vector in 3.27
    :return: the partial derivative of beta[j]
    '''
    m, n = np.shape(X)
    pd1 = np.zeros(n)
    for i in range(m):
        tmp = y[i] - sigmoid(X[i], beta)
        for j in range(n):
            pd1[j] += X[i][j] * tmp
    return pd1

def sigmoid(x, beta):
    '''
    :param x: is the predict variale
    :param beta: the parameter
    :return: the sigmoid function value
    '''
    return 1.0 / (1 + np.math.exp(- np.dot(beta, x.T)))

def gradDscent_1(X, y): # implementation of fundational gradDscent algorithms
    '''
    :param X: X is the variable matrix
    :param y: y is the label array
    :return: the best parameter estimate of 3.27
    '''
    import matplotlib.pyplot as plt
    h = 0.1 # step length of iterator
    max_times = 500
    m, n = np.shape(X)
    b = np.zeros((n, max_times)) # for show convergence curve of parameter
    beta = np.zeros(n)
    delta_beta = np.ones(n) * h
    llh = 0
    llh_temp = 0

    for i in range(max_times):
        beta_temp = beta.copy()
        for j in range(n):
            # for partial derivative
            beta[j] += delta_beta[j]
            llh_temp = likelihood(X, y, beta)
            delta_beta[j] = - h * (llh_temp - llh) / delta_beta[j]

            b[j, i] = beta[j]
            beta[j] = beta_temp[j]
        beta += delta_beta
        llh = likelihood(X, y, beta)
    t = np.arange(max_times)
    f2 = plt.figure(3)
    p1 = plt.subplot(311)
    p1.plot(t, b[0])
    plt.ylabel('w1')

    p2 = plt.subplot(312)
    p2.plot(t, b[1])
    plt.ylabel('w2')

    p3 = plt.subplot(313)
    p3.plot(t, b[2])
    plt.ylabel('b')
    plt.show()
    return beta


def gradDscent_2(X, y):  # implementation of stochastic gradDescent algorithms
    '''
    @param X: X is the variable matrix
    @param y: y is the label array
    @return: the best parameter estimate of 3.27
    '''

    import matplotlib.pyplot as plt
    m, n = np.shape(X)
    h = 0.5  # step length of iterator and initial
    beta = np.zeros(n)  # parameter and initial
    delta_beta = np.ones(n) * h
    llh = 0
    llh_temp = 0
    b = np.zeros((n, m))  # for show convergence curve of parameter
    for i in range(m):
        beta_temp = beta.copy()
        for j in range(n):
            # for partial derivative
            h = 0.5 * 1 / (1 + i + j)  # change step length of iterator
            beta[j] += delta_beta[j]
            b[j, i] = beta[j]
            llh_tmp = likelihood_sub(X[i], y[i], beta)
            delta_beta[j] = -h * (llh_tmp - llh) / delta_beta[j]
            beta[j] = beta_temp[j]
        beta += delta_beta
        llh = likelihood_sub(X[i], y[i], beta)
    t = np.arange(m)
    f2 = plt.figure(3)
    p1 = plt.subplot(311)
    p1.plot(t, b[0])
    plt.ylabel('w1')
    p2 = plt.subplot(312)
    p2.plot(t, b[1])
    plt.ylabel('w2')
    p3 = plt.subplot(313)
    p3.plot(t, b[2])
    plt.ylabel('b')
    plt.show()
    return beta

def predict(X, beta):
    '''
    prediction the class label using sigmoid
    @param X: data sample form like [x, 1]
    @param beta: the parameter of sigmoid form like [w, b]
    @return: the class label array
    '''
    m, n = np.shape(X)
    y = np.zeros(m)
    for i in range(m):
        if sigmoid(X[i], beta) > 0.5: y[i] = 1;
    return y

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.chdir(r"D:\7-学习资料\a-study\machinlearning\daima")
    data = pd.read_csv('watermelon.csv')
    dataset = data.iloc[:, 1:3].values

    pos = data[data['好瓜'] == '是'].index.tolist()
    df2 = pd.DataFrame(np.concatenate([np.ones([8, 1]), np.zeros([8, 1])]), columns=['labels'])
    data['labels'] = df2['labels'].apply(int)
    # y = data['labels']
    # dataset1 = dataset[dataset[:,0].argsort()] #按照密度大小值进行排序
    #
    # '''查看原始数据'''
    # plt.plot(dataset1[:,0],dataset1[:,1])
    # plt.show()
    '''查看数据集'''
    X = dataset
    y = data['labels']
    m, n = np.shape(X)
    f1 = plt.figure(1)
    plt.title('watermelon_3a')
    plt.xlabel('density')
    plt.ylabel('ratio_sugar')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=100, label='bad')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=100, label='good')
    plt.legend(loc='upper right')

    '''逻辑回归'''
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    # using gradDescent to get the optimal parameter beta = [w, b] in page-59
    beta = gradDscent_2(X_train, y_train)
    # prediction, beta mapping to the model
    y_pred = predict(X_test, beta)

    m_test = np.shape(X_test)[0]
    # calculation of confusion_matrix and prediction accuracy
    cfmat = np.zeros((2, 2))

    for i in range(m_test):
        if y_pred[i] == y_test[i] == 0:
            cfmat[0, 0] += 1
        elif y_pred[i] == y_test[i] == 1:
            cfmat[1, 1] += 1
        elif y_pred[i] == 0:
            cfmat[1, 0] += 1
        elif y_pred[i] == 1:
            cfmat[0, 1] += 1

    print(cfmat)