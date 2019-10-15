#-*- coding:utf-8 -*-
# @author: qianli
# @file: SK_DecisionTree.py
# @time: 2019/04/01

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
iris = load_iris()
data = iris.data[:, 2:]
target = iris.target
# tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf = DecisionTreeClassifier(criterion='entropy', max_depth=2)
tree_clf.fit(data, target)
# 估算类别
tree_clf.predict_proba([[5, 1.5]])
tree_clf.predict([[5,1.5]])
# export_graphviz(tree_clf, out_file=image_path("iris_tree.dot"), feature_names=iris.feature_names[2:],
#                 class_names=iris.target_names, rounded=True, filled=True)
#%%