#-*- coding:utf-8 -*-
# @author: qianli
# @file: DT_ID3_TEST.py
# @time: 2019/03/24
from sklearn.datasets import load_iris
import numpy as np
import math
from collections import Counter

'''
definition of decision node class
attr: attribution as parent for a new branching 
attr_down: dict: {key, value}
        key:   categoric:  categoric attr_value 
               continuous: '<= div_value' for small part
                           '> div_value' for big part
        value: children (Node class)
label： class label (the majority of current sample labels)
'''
class Node(object):
    def __init__(self, attr_init=None, label_init=None, attr_down_init={}):
        self.attr = attr_init
        self.label = label_init
        self.attr_down = attr_down_init
