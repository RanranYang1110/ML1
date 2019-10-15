#-*- coding:utf-8 -*-
# @author: qianli
# @file: test_dt.py
# @time: 2019/03/24

#%%

import pandas as pd
from collections import Counter
import numpy as np
def Entropy(p):
    return - p * np.log2(p)

from sklearn.datasets import load_iris
# trainData = [[0,0,0,0],[0,0,0,1],[0,1,0,1],[0,1,1,0],[0,0,0,0],
#              [1,0,0,0],[1,0,0,1],[1,1,1,1],[1,0,1,2],[1,0,1,2],
#              [2,0,1,2],[2,0,1,1],[2,1,0,1],[2,1,0,2],[2,0,0,0]]
# trainLabel = [0,0,1,1,0,0,0,1,1,1,1,1,1,1,0]
# def createDataSet():
#     dataSet = [['青年', '否', '否', '一般','拒绝'], ['青年', '否', '否', '好','拒绝'], ['青年', '是', '否', '好','同意'],
#                ['青年', '是', '是', '一般', '同意'], ['青年', '否', '否', '一般', '拒绝'], ['中年', '否', '否', '一般', '拒绝'],
#                ['中年', '否', '否', '好','拒绝'], ['中年', '是', '是', '好', '同意'], ['中年', '否', '是', '非常好', '同意'],
#                ['中年', '否', '是', '非常好','同意'], ['老年', '否', '是', '非常好','同意'], ['老年', '否', '是', '好','同意'],
#                ['老年', '是', '否', '好','同意'], ['老年', '是', '否', '非常好','同意'], ['老年', '否', '否', '一般','拒绝'],]
#     featureName = ['年龄','有工作','有房子','信贷情况']
#     return dataSet, featureName
class Node(object):
    def __init__(self, attr_init=None, label_init=None, attr_down_init={} ):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init

def calculate_Entropy(dataset):
    num = pd.Series(Counter(dataset)) #dataset的不重复数值及个数
    len_dataset = len(dataset)
    E = 0
    for i in range(num.shape[0]):
        p1 = num[i] / len_dataset
        E += - p1 * np.log2(p1) #计算信息熵
    return E

def GainEnt(X, y):
    #计算信息增益
    ent_X = calculate_Entropy(y) #首先计算总的信息熵
    num = pd.Series(Counter(y))
    labels = num.keys()
    labels_number = list(num)
    num_feature, num_class = X.shape
    Ev1 = []
    for i in range(num_class): #逐列计算每一列的信息增益
        data = X[X.columns[i]]
        num2 = pd.Series(Counter(data))  # dataset的不重复数值及个数
        data_labels = num2.keys() #获取每一列中不重复值
        E1 = 0
        for data_label in data_labels:
            pos = np.where(data == data_label)
            y1 = np.array(y)
            data_tree_label = y1[(pos[0])]
            num3 = pd.Series(Counter(data_tree_label))/len(data_tree_label) #计算各类指标的信息熵
            EV = 0
            for num3_class in range(len(num3)):
                num4 = list(num3)
                EV += - num4[num3_class] * np.log2(num4[num3_class])
            E1 += EV * num2[data_label] /len(data) #计算各类的信息熵和
        Ev1.append(E1)
        Ent_gain = ent_X - Ev1 #计算信息增益值
    return Ent_gain

def TreeGenerate(df):
    new_node = Node(None, None, {}) #用于储存决策树的信息
    y = df[df.columns[-1]]
    label_count = Counter(y) # label-y的种类
    if  label_count: #y不为空：
        new_node.label = max(label_count, key=label_count.get)
        if len(label_count) == 1 or len(y) == 0:
            return new_node

        X = df[df.columns[1:-1]]
        Ent_gain = GainEnt(X,y) #计算每一列的信息增益
        GainEnt_max_pos = np.where(Ent_gain == max(Ent_gain))[0][0] #判断信息增益最大的列
        GainEnt_max_name = X.columns[GainEnt_max_pos] # 得到信息增益最大对应的列
        new_node.attr = GainEnt_max_name
        num_class = pd.Series(Counter(df[GainEnt_max_name])) #信息增益最大列下的数据对应的不重复值
        for classname in num_class.keys():
            df_v = df[df[GainEnt_max_name].isin([classname])]
            df_v = df_v.drop(GainEnt_max_name, 1) #删除信息增益最大对应的列
            new_node.attr_down[classname] = TreeGenerate(df_v) #重复进行决策树的生成，y的种类为空
    return new_node

def GainEnt_max(X, y):
    Ent_gain = GainEnt(X, y)
    GainEnt_max_pos = np.where(Ent_gain == max(Ent_gain))[0][0]
    return GainEnt_max_pos

def predict(root, df_sample):
    try :
        import re
    except ImportError:
        print('module re not found')
    while root.attr != None:
        key = df_sample[root.attr].values[0]
        if key in root.attr_down:
            root = root.attr_down[key]
        else:
            break
    return root.label

if '__name__' == '__main__':
    import os
    from random import sample
    os.chdir(r"D:\7-学习资料\a-study\python\Machine-Learning_ZhouZhihua-master\ch4_decision_tree\4.3_ID3\data")
    data_file_encode = "gb18030"
    with open('watermelon_3.csv', mode='r', encoding=data_file_encode) as f:
        df = pd.read_csv(f)
    df = df.drop(['密度', '含糖率'], 1) #把密度和含糖率对应的列去除掉
    root = TreeGenerate(df)
    accuracy_scores = []

    for i in range(10):
        train = sample(range(len(df.index)), int(1 * len(df.index) / 2))
        df_train = df.iloc[train]
        df_test = df.drop(train)
        # generate the tree
        root = TreeGenerate(df_train)
        # test the accuracy
        pred_true = 0
        for i in df_test.index:
            label = predict(root, df[df.index == i])
            if label == df_test[df_test.columns[-1]][i]:
                pred_true += 1

        accuracy = pred_true / len(df_test.index)
        accuracy_scores.append(accuracy)