'''模型实例化-导入VGG16模型'''
from keras.applications import VGG16
from keras import optimizers
from keras import models
from keras import layers
from keras.preprocessing.image import ImageDataGenerator
# 原始数据集路径位置
train_dir = '/home/qianli/DL/01 data/00 case_bear_keras_clf/train'
test_dir = '/home/qianli/DL/01 data/00 case_bear_keras_clf/test'
validation_dir = '/home/qianli/DL/01 data/00 case_bear_keras_clf/validation'
original_dataset_dir = '/home/qianli/DL/01 data/00 case_bear_keras_clf/train'

# 数据增强
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(256,256),
                                                    batch_size=20, class_mode='categorical')
validata_generator = test_datagen.flow_from_directory(validation_dir, target_size=(256,256),
                                                      batch_size=20, class_mode='categorical')
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape', labels_batch.shape)
    break

'''建立模型'''
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(256,256,3))
conv_base.summary()

'''在卷积基上添加一个密集连接分类器'''
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))
model.summary()

#在编译和训练模型之前，一定要“冻结”卷积基。冻结一个或多个层是指在训练过程中保持其权重不变。
#如果不这么做，卷积基之前学到的表示将会在训练过程中被修改，对之前学到的表示造成很大的破坏。
#冻结方法：将trainable转成False
'''冻结卷积基'''
print('This is the number of trainable weights '
'before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights '
'after freezing the conv base:', len(model.trainable_weights))

#卷积基冻结之后，仅新添加的Dense层的权重会被训练。总共有4个权重张量，每层有2个。
'''编译模型'''
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='categorical_crossentropy', metrics=['acc'])

'''训练模型'''
history = model.fit_generator(train_generator,steps_per_epoch=100, epochs=30,
                              validation_data=validata_generator, validation_steps=50)
'''模型保存'''
model.save('/home/qianli/DL/02 model/bearsDiag_vgg16_2.h5')

'''保存数据'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
import pandas as pd
import numpy as np
result = np.c_[epochs, acc, val_acc, loss, val_loss]
result_dataframe = pd.DataFrame(result, columns=['epochs','acc','val_acc',
                                                 'loss','val_loss'])
result_dataframe.to_csv('/home/qianli/DL/02 model/result1.csv')
# plt.plot(epochs, acc, 'bo', label = 'Training acc')
# plt.plot(epochs, val_acc, 'b', label ='Validation acc')
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.show()
#
# plt.figure(2)
# plt.plot(epochs, loss, 'bo', label = 'Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.legend()
# plt.show()


