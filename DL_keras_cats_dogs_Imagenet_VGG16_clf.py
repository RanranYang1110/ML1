import numpy as np

'''使用在ImageNet上训练的VGG16网络的卷积层，从猫狗图像中提取有趣的特征，然后在
这些特征上训练一个猫狗分类器'''

'''模型实例化-导入VGG16模型'''
from keras.applications import VGG16
from keras import optimizers
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150,150,3))
#weights 指定模型初始化的权重检查点
# include_top 指定模型最后是否包含密集连接分类器
# input_shape 输入到网络中的图像张量的形状
conv_base.summary()

'''不使用数据增强的快速特征提取'''
from keras.preprocessing.image import ImageDataGenerator

train_dir = r'D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\train'
test_dir = r'D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\test'
validation_dir = r'D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\validation'
original_dataset_dir = r'D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\train'
train_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\train\dogs'
train_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\train\cats'
validation_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\validation\dogs'
validation_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\validation\cats'
test_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test\dogs'
test_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test\cats'

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 20


'''在卷积基上添加一个密集连接分类器'''
from keras import models
from keras import layers
model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
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
model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss='binary_crossentropy', metrics=['acc'])
'''利用冻结的卷积基端到端地训练模型'''
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
train_datagen = ImageDataGenerator(rescale=1./255, height_shift_range=0.2, width_shift_range=0.2,
                             rotation_range=40, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,
                             fill_mode='nearest')
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150,150),class_mode='binary',
                                                    batch_size=20)
validata_generator = test_datagen.flow_from_directory(validation_dir,target_size=(150,150),class_mode='binary',
                                                    batch_size=20)
history = model.fit_generator(train_generator,steps_per_epoch=100, epochs=30,
                              validation_data=validata_generator, validation_steps=50)
model.save(r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\cats_and_dogs_small_vgg16_1.h5')
'''绘制结果'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label = 'Training acc')
plt.plot(epochs, val_acc, 'b', label ='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(epochs, loss, 'bo', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()