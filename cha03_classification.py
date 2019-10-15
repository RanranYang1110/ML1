import warnings
warnings.filterwarnings("ignore")
'''使用keras导入mnist数据'''
from keras.datasets import mnist
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# '''使用tensorflow导入mnist数据'''
# from tensorflow.examples.tutorials.mnist import  input_data
# mnist = input_data.read_data_sets('D:/5-旋转设备无忧运行系统/2-深度学习轴承故障诊断/MNIST_data/',one_hot = True)
#
#
# print(mnist.train.images.shape)
# print(mnist.train.labels.shape)

'''获取训练集和测试集'''
# X_train = mnist.train.images
# Y_train = mnist.train.labels
# X_test = mnist.test.images
# Y_test = mnist.test.labels

'''可视化'''
import matplotlib.pyplot as plt
import matplotlib
some_digit = X_train[36000,:]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation = 'nearest')
plt.axis('off')
plt.show()
Y_train[36000]

'''训练集数据洗牌'''
import numpy as np
shuffle_index = np.random.permutation(55000)
X_train, Y_train = X_train[shuffle_index], Y_train[shuffle_index]

'''训练一个二元分类器'''
y_train_5 = (Y_train == 5)
y_test_5 = (Y_test == 5)


#创建一个随机梯度下降（SGD）分类器
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
#检测数字5的图像
sgd_clf.predict([some_digit])

'''性能评估-交叉验证'''
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, random_state=42)
for train_index, test_index in skfolds.split(X_train,y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = (y_train_5[train_index])
    X_test_fold = X_train[test_index]
    y_test_fold = (y_train_5[test_index])
    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred))

'''cross_val_score评估SGDClassifier模型'''
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy')

'''混淆矩阵'''
from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# cross_val_predict()函数返回的是每个折叠的预测
# confusion_matrix()函数获取混淆矩阵
from sklearn.metrics import confusion_matrix
confusion_matrix(y_train_5, y_train_pred)
# 混淆矩阵行表示实际类别，列表示预测类别

'''计算分类器精度和召回率'''
from sklearn.metrics import precision_score,recall_score
precision_score(y_train_5, y_train_pred)
recall_score(y_train_5,y_train_pred)

'''计算F1分数'''
from sklearn.metrics import  f1_score
f1_score(y_train_5, y_train_pred)

'''精度/召回率权衡'''
#sklearn可用于访问它用于预测的决策分数
y_scores = sgd_clf.decision_function([some_digit])
threshold = 0
y_some_digit_pred = (y_scores > threshold)
#SGDClassifier分类器使用的阈值为0

'''阈值确定方法'''
# 使用cross_val_predict()获取训练集种所有实例的分数
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
# 使用precision_recall_curve()来计算所有可能的阈值的精确和召回率
from sklearn.metrics import  precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

'''多类别分类器-sgd分类'''
sgd_clf.fit(X_train, Y_train)
sgd_clf.predict([some_digit])

some_digit_scores = sgd_clf.decision_function([some_digit])
np.argmax(some_digit_scores)

'''基于SGDClassifier 创建一个多类别分类器'''
from sklearn.multiclass import OneVsOneClassifier
ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
ovo_clf.fit(X_train, Y_train)
ovo_clf.predict([some_digit])
len(ovo_clf.estimators_)

'''训练一个随机森林分类器'''
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(random_state=42)
forest_clf.fit(X_train, Y_train)
forest_clf.predict([some_digit])
'''分类器将每个实例分类为每个类别的概率列表'''
forest_clf.predict_proba([some_digit])

'''交叉验证评估SGDClassifier的准确率'''
cross_val_score(sgd_clf, X_train, Y_train, cv = 3, scoring='accuracy')

'''查看混淆矩阵'''
y_train_pred = cross_val_predict(sgd_clf, X_train, Y_train, cv=3)
conf_mx = confusion_matrix(Y_train, y_train_pred)
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()
#将混淆矩阵中的每个值除以相应类别中的图片数量
row_sums = conf_mx.sum(axis=1, keepdims = True)
norm_conf_mx = conf_mx/row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
plt.show()

'''多标签分类系统'''
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (Y_train >= 7)
y_train_odd = (Y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train,y_multilabel)
knn_clf.predict([some_digit])

#%%
# 计算所以标签的平均F1分数来评估多标签分类器
# y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
# f1_score(y_multilabel, y_train_knn_pred, average='macro')

'''给MNIST图片增加噪声'''
noise = np.random.randint(0,100,(len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0,100,(len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")

some_index = 5500
plt.subplot(121)
plot_digit(X_test_mod[some_index])
plt.subplot(122)
plot_digit(y_test_mod[some_index])
plt.show()

'''通过训练分类器，清晰图片'''
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)