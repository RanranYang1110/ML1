#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_lstm2.py
# @time: 2019/06/23
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir(r"D:\7-学习资料\a-study\datasets")
path = 'jena_climate_2009_2016.csv'
with open(path) as f:
    data = pd.read_csv(f).values
data = data[:,1:]
data = data.astype('float32')
temp = data[:,1]
plt.plot(range(len(temp)), temp)
plt.plot(range(1440), temp[:1440]) # 查看前10天数据

lookback = 720 #给定过去5天内的观测数据
steps = 6 #观测数据的采样频率是每小时一个数据点
delay = 144 #目标是未来24h之后的数据
#%%
'''数据标准化'''
mean = data[:200000].mean(axis=0)
data1 = data - mean
std = data1[:200000].std(axis=0)
data2 = data1 / std

'''生成器，生成一个元组（samples, targets）,samples是输入数据的一个批量，targets是对应的目标温度数组
生成器的参数：
data:标准化的原始数组
lookback: 输入数据应该包括过去多少个时间步
delay: 目标是未来多少个时间步之后
min_index, max_index: data数组中的索引，用于界定需要抽取哪些时间步。
shuffle:打乱样本，或按顺序抽取样本
batch_size：每个批量的样本数
step: 数据采样的周期
'''

def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

'''准备训练生成器、验证生成器和测试生成器'''
lookback = 1440
step = 6
delay = 144
batch_size = 128
train_gen = generator(data2, lookback=lookback, delay=delay, min_index=0, max_index=200000,
                      shuffle=True, step=step, batch_size=batch_size)
val_gen = generator(data2, lookback=lookback, delay=delay, min_index=200001, max_index=300000,
                      step=step, batch_size=batch_size)
test_gen = generator(data2, lookback=lookback, delay=delay, min_index=300001, max_index=None,
                      step=step, batch_size=batch_size)
val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(data2) - 300001 - lookback) // batch_size

'''计算符合常识的基准方法MAE'''
#始终预测24小时后的温度等于现在的温度
def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()

'''一种基本的机器学习方法(小型的密集连接网络)'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, data2.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)
'''绘制结果'''
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''训练并评估一个基于GRU的模型'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, input_shape=(None, data2.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=20,
                              validation_data=val_gen, validation_steps=val_steps)

'''绘制结果'''
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

'''训练并评估一个使用dropout正则化的基于GRU的模型'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, dropout=0.2, recurrent_dropout=0.2, input_shape=(None, data2.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=40,
                              validation_data=val_gen, validation_steps=val_steps)

'''训练并评估一个使用dropout正则化的堆叠GRU模型'''
from keras import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.GRU(32, dropout=0.1, recurrent_dropout=0.5,
                     return_sequences=True, input_shape=(None, data2.shape[-1])))
model.add(layers.GRU(64, activation='relu', dropout=0.1, recurrent_dropout=0.5))

'''使用逆序序列训练并评估一个LSTM'''
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras import layers
from keras.models import Sequential
max_features = 10000
maxlen = 500
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features) #加载数据
x_train = [x[::-1] for x in x_train]
x_test = [x[::-1] for x in x_test] #将序列反转
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
model = Sequential()
model.add(layers.Embedding(max_features, 128))
model.add(layers.LSTM(32))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

'''训练并评估一个双向LSTM'''
model = Sequential()
model.add(layers.Embedding(max_features, 32))
model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.2)

'''训练一个双向GRU'''
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop
model = Sequential()
model.add(layers.Bidirectional(layers.GRU(32), input_shape=(None, data2.shape[-1])))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(), loss='mae')
history = model.fit_generator(train_gen, steps_per_epoch=500, epochs=10,
                              validation_data=val_gen, validation_steps=val_steps)
model.predict(test_gen, batch_size=128)