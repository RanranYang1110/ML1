'''加载IMDB(互联网电影数据库)数据'''
from keras.datasets import imdb
(train_data, train_labels),(test_data, test_labels) = imdb.load_data(num_words=10000)
# num_words是只保留训练数据中前10000个最常出现的单词
train_data[0]
train_labels[0]
'''将某条评论迅速解码为英文单词'''
word_index = imdb.get_word_index() #word_index是一个将单词映射为整数索引的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decoded_review = ' '.join(reverse_word_index.get(i - 3, '?') for i in train_data[0])

'''将整数序列编码为二进制矩阵'''
import numpy as np
def vectorize_sequences(sequences, dimension = 10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.0
    return results
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

'''标签向量化'''
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

'''构建网络'''
from keras import models,layers
model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

'''配置优化器'''
from keras import optimizers
model.compile(optimizer= optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

'''使用自定义的损失和指标函数'''
from keras import losses
from keras import metrics
model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=['acc'])

'''留出验证集'''
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]
'''训练模型'''
history = model.fit(partial_x_train, partial_y_train, batch_size=512, epochs=4, validation_data=(x_val, y_val))
history_dict = history.history
history_dict.keys()
'''绘制训练损失和验证损失'''
import matplotlib.pyplot as plt
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

'''绘制训练精度和验证激精度'''
plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']
epochs = range(1, len(loss_values)+1)
plt.plot(epochs, acc_values, 'bo', label='Training acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

results = model.evaluate(x_test, y_test)
y_test_pred = model.predict(x_test)