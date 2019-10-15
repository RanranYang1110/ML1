#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_bp111.py
# @time: 2019/03/26
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import numpy as np
from sklearn.preprocessing import LabelBinarizer #标签二值化

class BP(object):
    def __init__(self, layers):
        self.v = np.random.random((layers[0]+1, layers[1])) * 2 -1 #(65,100)
        self.w = np.random.random((layers[1], layers[2]))* 2 -1 #(100,10)
    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))
    def dsigmoid(self, x):
        return x * (1 - x)
    def train(self, X, y, lr=0.2, epochs=10000):
        temp = np.ones([X.shape[0], X.shape[1]+1])
        temp[:, :-1]= X
        X = temp

        #权值训练更新
        for i in range(epochs+1):
            m = np.random.randint(X.shape[0])
            x = X[m,:].reshape(1, -1) # (1,65)
            L1 = self.sigmoid(np.dot(x, self.v)) #(1,100)
            L2 = self.sigmoid(np.dot(L1,self.w)) #(1,10)

            L2_delta = (y[m] - L2) * self.dsigmoid(L2) # (1,10)
            L1_delta = L2_delta.dot(self.w.T) * self.dsigmoid(L1) #(1,100)

            self.w += lr * L1.T.dot(L2_delta) #(100,10)
            self.v += lr * x.T.dot(L1_delta) #(65,100)
            if i % 1000 == 0:
                predictions = []
                for j in range(X_test.shape[0]):
                    out = self.predict(X_test[j,:])
                    predictions.append(np.argmax(out))
                accuracy = np.mean(np.equal(predictions, y_test))
                print('epoch:', i, 'accuracy:', accuracy)
    def predict(self,x):
        temp = np.ones(x.shape[0] + 1)
        temp[:-1] = x
        x = temp.reshape(1,-1)
        L1 = self.sigmoid(np.dot(x, self.v))  # (1,100)
        L2 = self.sigmoid(np.dot(L1, self.w))  # (1,10)
        return L2

digits = load_digits()
data = digits.data
target = digits.target
data -= data.min()
data /= data.max()
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)
#标签二值化
labels_train = LabelBinarizer().fit_transform(y_train)
print(labels_train[0:10])
labels_test = LabelBinarizer().fit_transform(y_test)
print('start')
nm = BP([64,100,10])
nm.train(X_train, labels_train, epochs=20000)
print('end')