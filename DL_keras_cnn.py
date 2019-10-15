from keras import models
from keras import layers
'''实例化一个小型的卷积神经网络'''
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(28, 28, 1),padding='same'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.summary()

'''在卷积神经网络上添加分类器'''
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()

'''在mnist图像上训练卷积神经网络'''
from keras.datasets import mnist
(train_data, train_labels),(test_data, test_labels) = mnist.load_data()
train_images = train_data.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255
test_images = test_data.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

from keras.utils import to_categorical
train_labels = to_categorical(train_labels,10)
test_labels = to_categorical(test_labels, 10)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_images, train_labels, batch_size=512, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
