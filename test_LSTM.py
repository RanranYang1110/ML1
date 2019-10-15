#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_LSTM.py
# @time: 2019/06/23

'''简单RNN的numpy实现'''
import numpy as np
timesteps = 100 # 输入序列的时间步数
input_features = 32 #输入特征空间的维度
output_features = 64 #输出特征空间的维度
inputs = np.random.random((timesteps, input_features)) #输入数据：随机噪声，仅作为示例
state_t = np.zeros((output_features, )) #初始状态：全零向量
W = np.random.random((output_features, input_features)) #创建随机的权重矩阵
U = np.random.random((output_features, output_features))
b = np.random.random((output_features, ))
successive_outputs = []
for input_t in inputs: #input_t是形状为（input_features,）的向量
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.stack(successive_outputs, axis=0)

from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.summary()
# SimpleRNN 可以在两种不同的模式下运行：一种是返回每个时间步连续输出的完整序列，
# 即形状为（batch_size, timesteps, output_features）的三维张量；
# 一种是只返回每个输入序列的最终输出，即形状为（batch_size, output_features）的二维张量
# return_sequencesd可控制这两种模式

'''为了提高网络的表达能力，将多个循环层逐个堆叠有时也是很有用的'''
model = Sequential()
model.add(Embedding(10000, 32))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))
model.summary()

'''准备IMDB数据'''
from keras.datasets import imdb
from keras.preprocessing import sequence
max_features = 10000 #作为特征的单词个数
maxlen = 500
batch_size = 32
print('Loading data ...')
(input_train, y_train),(input_test, y_test)=imdb.load_data(num_words=max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

'''用Embeding层和SimpleRNN来训练模型'''
from keras.layers import Dense
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

'''绘制结果'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.show()

'''Keras中一个LSTM的具体例子'''
from keras.layers import LSTM
model = Sequential()
model.add(Embedding(max_features, 32))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train, epochs=10, batch_size=128, validation_split=0.2)
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.show()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.show()