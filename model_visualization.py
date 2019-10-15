from keras.models import load_model
model = load_model(r"D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\cats_and_dogs_small_3.h5")
model.summary()

img_path = r"D:\7-学习资料\a-study\datasets\dog_cat\01 cats_and_dogs_small\test\dogs\dog.1500.jpg"
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)
model.predict(img_tensor)
'''显示测试图像'''
import matplotlib.pyplot as plt
plt.imshow(img_tensor[0])
plt.show()
'''用一个输入张量和一个输出张量列表将模型实例化'''
from keras import models
layer_outputs = [layer.output for layer in model.layers[:12]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#创建一个模型，给定模型输入，可以返回这些输出
activations = activation_model.predict(img_tensor) #返回4个numpy数组组成的列表，每个层激活对应一个numpy数组
'''第一个卷积层的激活情况'''
first_layer_activation = activations[0]
print(first_layer_activation.shape)
'''第四个通道可视化'''
import matplotlib.pyplot as plt
plt.matshow(first_layer_activation[0, :, :, 6],cmap='viridis')
'''将每个中间激活的所有通道可视化'''
layer_names = []
for layer in model.layers[:12]:
    layer_names.append(layer.name)
images_per_row = 16
for layer_name, layer_activation in zip(layer_names,activations):
    n_features = layer_activation.shape[-1] #特征图中的特征个数
    size = layer_activation.shape[1] #特征图的形状为（1,size, size, n_features）
    n_cols = n_features // images_per_row #在矩阵中将激活通道平铺
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:,:,col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
            row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
''''''
from keras.models import load_model
model = load_model(r"D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\model\bearsDiag_vgg16_1.h5")
model.summary()

img_path = r"D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\test\1-or\12_de_X130_DE_time4.jpg"
from keras.preprocessing import image
import numpy as np
img = image.load_img(img_path, target_size=(256,256))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)
pre = model.predict(img_tensor)
model.predict(img)

#%%
test_dir = r'D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\figdata\00 case_bear_keras_clf\test'

from keras.preprocessing.image import ImageDataGenerator
# 数据增强
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=324, target_size=(256,256),class_mode='categorical')
test_datas, test_labels = test_generator.next()
# pre = model.estimate(test_generator)
pre = model.predict(test_datas)
#%%
pre1 = pre.T
pos = np.where(pre1 == np.max(pre1,axis=0))
test_pre = pos[0]
plt.plot(test_pre)
#%%
test_y = np.where(test_labels == np.max(test_labels,axis=0))
test_y = test_y[1]
plt.plot(test_y)
#%%
loss, acc = model.evaluate(test_datas, test_labels)
#%%
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=324, target_size=(256,256),class_mode='categorical')
# test_datas, test_labels = test_generator.next()
bottleneck_features_validation = model.predict_generator(test_generator, 324)
