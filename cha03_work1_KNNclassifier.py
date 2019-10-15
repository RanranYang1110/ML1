from sklearn.neighbors import KNeighborsClassifier
from keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score

'''使用keras导入mnist数据'''
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

'''训练KNN模型，预测结果'''
knn_clf = KNeighborsClassifier(n_jobs=-1, weights='distance', n_neighbors=4)
# cross_val_score(knn_clf, X_train, Y_train, cv=3, scoring='accuracy')
knn_clf.fit(X_train,Y_train)
y_knn_pred = knn_clf.predict(X_test)

'''查看结果'''
conf_mx = confusion_matrix(Y_test, y_knn_pred)
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()
#将混淆矩阵中的每个值除以相应类别中的图片数量
row_sums = conf_mx.sum(axis=1, keepdims = True)
norm_conf_mx = conf_mx/row_sums
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap = plt.cm.gray)
plt.show()

'''查看结果准确率'''
accuracy_score(y_knn_pred,Y_test)

'''实现MNIST图片想任意方向移动'''
from scipy.ndimage.interpolation import shift
def shift_digit(digit_array, dx, dy, new = 0):
    return shift(digit_array.reshape(28,28), [dy, dx], cval=new).reshape(784)

def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.binary,interpolation="nearest")
    plt.axis("off")

X_train_expanded = [X_train]
y_train_expanded = [Y_train]
for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
    shifted_images = np.apply_along_axis(shift_digit, axis=1, arr=X_train, dx=dx, dy=dy)
    X_train_expanded.append(shifted_images)
    y_train_expanded.append(Y_train)
X_train_expanded = np.concatenate(X_train_expanded)
y_train_expanded = np.concatenate(y_train_expanded)
X_train_expanded.shape, y_train_expanded.shape

knn_clf.fit(X_train_expanded,y_train_expanded)
y_knn_expanded_pred = knn_clf.predict(X_test)
accuracy_score(Y_test, y_knn_expanded_pred)