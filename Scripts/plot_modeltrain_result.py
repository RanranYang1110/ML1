#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.chdir(r"D:\5-旋转设备无忧运行系统\2-深度学习轴承故障诊断\00 AIMS轴承故障分类\00data\00 AIMS_bear_keras_4712_de\model")
data = pd.read_csv("result1.csv")
epochs = data['epochs']
acc = data['acc']
val_acc = data['val_acc']
loss = data['loss']
val_loss = data['val_loss']

#%%
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
