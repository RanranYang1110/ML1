#-*- coding:utf-8 -*-
# @author: qianli
# @file: sk_RandomForest.py
# @time: 2019/03/31
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
iris = load_iris()
data = iris.data
target = iris.target
X_train, X_test, y_train, y_test = train_test_split(data,target,test_size=0.2)
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
clf.predict(X_test)
#%%
clf.predict_proba(X_test)
list(zip(iris.feature_names, clf.feature_importances_))