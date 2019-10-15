#-*- coding:utf-8 -*-
# @author: qianli
# @file: Bayes_diy.py
# @time: 2019/03/29

import numpy as np
import pandas as pd
import os
from collections import Counter
'''导入西瓜数据集'''
# os.chdir(r"D:\7-学习资料\a-study\python\Machine-Learning_ZhouZhihua-master\ch4_decision_tree\4.3_ID3\data")
# data_file_encode = "gb18030"
# with open('watermelon_3.csv', mode='r', encoding=data_file_encode) as f:
#     df = pd.read_csv(f)
# df = df.drop(['密度', '含糖率'],1)
# X = df[df.columns[:-1]]
# y = df[df.columns[-1]]
'''导入水果数据集'''
datasets = {'banala':{'long':400,'not_long':100,'sweet':350,'not_sweet':150,'yellow':450,'not_yellow':50},
            'orange':{'long':0,'not_long':300,'sweet':150,'not_sweet':150,'yellow':300,'not_yellow':0},
            'other_fruit':{'long':100,'not_long':100,'sweet':150,'not_sweet':50,'yellow':50,'not_yellow':150}}
def count_total(data):
    '''计算各种水果的总数'''
    count = {}
    total = 0
    for fruit in data:
        count[fruit] = data[fruit]['sweet'] + data[fruit]['not_sweet']
        total += count[fruit]
    return count, total
count, total = count_total(datasets)
def cal_base_rates(data):
    '''计算各种水果的先验概率'''
    categories, total = count_total(data)
    cal_base_rates = {}
    for label in categories:
        priori_prob = categories[label]/total
        cal_base_rates[label] = priori_prob
    return cal_base_rates
cal_base_rates = cal_base_rates(datasets)

def likelihold_prob(data):
    '''计算各个特征值在已知水果下的概率（likelihood probabilities）
    {'banala':{'long':0.8}...}'''
    count, _ = count_total(data)
    likelihold = {}
    for fruit in data:
        '''创建一个临时的字典，临时存储各个特征值的概率'''
        attr_prob = {}
        for attr in data[fruit]:
            attr_prob[attr] = data[fruit][attr]/count[fruit]
        likelihold[fruit] = attr_prob
    return likelihold
# likelihold = likelihold_prob(datasets)
# print(likelihold)
def evidence_prob(data):
    '''计算特征的概率对分类结果的影响
    #水果的所有特征'''
    attrs = list(data['banala'].keys())
    count, total = count_total(data)
    evidence_prob = {}
    #计算各种特征的概率
    for attr in attrs:
        attr_total = 0
        for fruit in data:
            attr_total += data[fruit][attr]
        evidence_prob[attr] = attr_total/total
    return evidence_prob
Evidence_prob = evidence_prob(datasets)
print(Evidence_prob)
#%%