from keras.datasets import boston_housing
#%%
(train_data, train_targets),(test_data, test_targets) = boston_housing.load_data()
train_data.shape

'''数据标准化'''
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

'''模型定义'''
from keras import models
from keras import layers
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

'''K折验证'''
import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([
        train_data[: i * num_val_samples],
        train_data[(i + 1) * num_val_samples :]],
        axis = 0)
    partial_train_targets = np.concatenate([
        train_targets[: i * num_val_samples],
        train_targets[(i + 1) * num_val_samples :]],
        axis = 0)
    model = build_model()
    model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
              batch_size=1, verbose=0) # verbose=0表示训练模型为静默模式
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

'''保存每折的验证结果'''
import numpy as np
k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_mae_histories = []
for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples : (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples : (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([
        train_data[: i * num_val_samples],
        train_data[(i + 1) * num_val_samples :]],
        axis = 0)
    partial_train_targets = np.concatenate([
        train_targets[: i * num_val_samples],
        train_targets[(i + 1) * num_val_samples :]],
        axis = 0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=num_epochs,
                        batch_size=1, verbose=0, validation_data=(val_data, val_targets)) # verbose=0表示训练模型为静默模式
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)
    '''计算所有轮次中的K折验证分数平均值'''
    average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
    '''绘制验证分数'''
    import matplotlib.pyplot as plt
    plt.plot(range(1,len(average_mae_history)+1),average_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()

    '''绘制验证分数（删除前10个数据点）'''


    def smooth_curve(points, factor=0.9):
        smoothed_points = []
        for point in points:
            if smoothed_points:
                previous = smoothed_points[-1]
                smoothed_points.append(previous * factor + point * (1 - factor))
            else:
                smoothed_points.append(point)
        return smoothed_points
    smooth_mae_history = smooth_curve(average_mae_history[10:])
    plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
    plt.xlabel('Epochs')
    plt.ylabel('Validation MAE')
    plt.show()
'''训练最终模型'''
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=1, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score

'''向模型添加L2权重正则化'''
from keras import regularizers
model = models.Sequential()
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001),
                       activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
# l2(0.001)是指该层权重矩阵的每个系数都会使网络总损失增加0.001*weight_coefficient_value
# 由于这个惩罚项只是在训练时添加，所以网络的训练损失会高于测试损失

'''keras中不同的权重正则化项'''
from keras import regularizers
regularizers.l1(0.001)
regularizers.l1_l2(l1=0.001, l2=0.001) #同时做L1和L2正则化