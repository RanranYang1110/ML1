import os,shutil
train_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\train'
test_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test'
validation_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\validation'
original_dataset_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\00 cats_and_dogs_alldata\train'
train_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\train\dogs'
train_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\train\cats'
validation_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\validation\dogs'
validation_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\validation\cats'
test_dogs_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test\dogs'
test_cats_dir = r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test\cats'

'''查看每个分组中分布包含多少图像'''
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total testing cat images', len(os.listdir(test_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total testing dog images', len(os.listdir(test_dogs_dir)))

'''构建网络'''
from keras import layers
from keras import models
model = models.Sequential()
'''利用ImageDataGenerator来设置数据增强'''
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
    shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
'''显示几个随机增强后的训练图像'''
from keras.preprocessing import image
import matplotlib.pyplot as plt
# fnames = os.listdir(train_cats_dir)
fnames = [os.path.join(train_cats_dir, fname) for fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150,150))
x = image.img_to_array(img) #形状转化为（150，150，3）的numpy数组
x = x.reshape((1,)+ x.shape) #将形状改变为(1,150,150,3)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()

'''在数据增强的基础上，添加dropout层，定义一个新的卷积神经网络'''
from keras.optimizers import RMSprop
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['acc'])

'''利用数据增强生成器训练卷积神经网络'''
#注意验证函数不能增强
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rotation_range=40, width_shift_range=0.2, height_shift_range=0.2,
                                   rescale = 1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale= 1./255)
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), batch_size=32, class_mode='binary')
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150),
                                                        batch_size=32, class_mode='binary')
history = model.fit_generator(train_generator,steps_per_epoch=100, epochs=100, validation_data=validation_generator,
                    validation_steps=50)
model.save(r'D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\cats_and_dogs_small_3.h5')

'''绘制训练过程中的损失曲线和精度曲线'''
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.figure(2)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()