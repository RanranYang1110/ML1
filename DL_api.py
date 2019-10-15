#-*- coding:utf-8 -*-
# @author: qianli
# @file: DL_api.py
# @time: 2019/07/22
from keras import Input, layers
input_tensor = Input(shape=(32,)) #一个张量
dense = layers.Dense(32, activation='relu') #一个层是一个函数
output_tensor = dense(input_tensor) #可以在一个张量上调用一个层，它会返回一个张量

'''一个最简单的示例，并列展示一个简单的sequential模型'''
from keras.models import Sequential, Model
from keras import layers
from keras import Input
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
'''对应的函数式API实现'''
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)

model = Model(input_tensor, output_tensor)
#model类将输入张量和输出张量转换为一个模型
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy') #编译模型
import numpy as np
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128) #训练10轮模型
score = model.evaluate(x_train, y_train) #评估模型

'''多输入模型'''
from keras.models import Model
from keras import layers
from keras import Input
'''文本输入是一个长度可变的整数序列。注意，你可以选择对输入进行命名'''
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input) #将输入嵌入长度为64的向量
encoded_text = layers.LSTM(32)(embedded_text) #利用LSTM将向量编码为单个向量
question_input = Input(shape=(None,), dtype='int32', name='question')
#对问题提进行相同的处理（使用不同的层实例）

embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)

concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1) #将编码后的问题和文本连接起来
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)


