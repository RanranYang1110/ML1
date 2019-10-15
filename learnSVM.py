#-*- coding:utf-8 -*-
# @author: qianli
# @file: learnSVM.py
# @time: 2019/10/15

import pandas as pd
import numpy as np
import sklearn as sk
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from sklearn.svm import SVC, LinearSVC

linear_svc = SVC(kernel='linear')
print(linear_svc)