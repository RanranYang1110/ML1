#-*- coding:utf-8 -*-
# @author: qianli
# @file: learn_hyperopt.py
# @time: 2019/08/02
from hyperopt import fmin, tpe, hp
best = fmin(
    # fn = lambda x: x,  # fmin接受一个函数的最小化，记为fn.
    fn = lambda x: (x-1) ** 2,
    space = hp.uniform('x', 0, 1),
    algo = tpe.suggest,
    max_evals = 100)
print(best)
#%%
from hyperopt import hp
import hyperopt.pyll.stochastic
space = {
    'x': hp.uniform('x', 0, 1),
    'y': hp.normal('y', 0, 1),
    'name': hp.choice('name', ['alice','bob']),
        }
print(hyperopt.pyll.stochastic.sample(space))

#%% Trials
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import matplotlib.pyplot as plt
fspace = {'x': hp.uniform('x', -5, 5)}
def f(params):
    x = params['x']
    val = x ** 2
    return {'loss':val, 'status': STATUS_OK}
trials = Trials()
best = fmin(fn=f, space=fspace, algo=tpe.suggest, max_evals=100, trials=trials)
print('best:', best)
print('trials:')
for trial in trials.trials[:2]:
    print(trial)

f, ax = plt.subplots(1)
xs = [t['misc']['vals']['x'] for t in trials.trials]
ys = [t['result']['loss'] for t in trials.trials]
# ax.set_xlim(xs[0]-10, xs[-1]+10)
ax.scatter(xs, ys, s=20, linewidth=0.01, alpha=0.75)
ax.set_title('$x$ $vs$ $t$ ', fontsize=18)
ax.set_xlabel('$t$', fontsize=16)
ax.set_ylabel('$x$', fontsize=16)
#%%
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from hyperopt import hp, STATUS_OK, Trials, fmin, tpe
import matplotlib.pyplot as plt
iris = load_iris()
X, y = iris.data, iris.target

def hyperopt_train(params):
    clf = KNeighborsClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space_knn = {'n_neighbors': hp.choice('n_neighbors', range(1, 100))}
def f(params):
    acc = hyperopt_train(params)
    return {'loss': -acc, 'status': STATUS_OK}
trials = Trials()
best = fmin(fn=f, space=space_knn, algo=tpe.suggest, max_evals=100, trials=trials)
print('best',best)

#%%
'''支持向量机参数优化'''
from sklearn.datasets import load_iris
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from hyperopt import fmin, hp, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
import numpy as np
iris = load_iris()
X = iris.data
y = iris.target

def hyperopt_train_test(params):
    clf = SVC(**params)
    return cross_val_score(clf, X, y).mean()

space_svm = {
    'C' : hp.uniform('C', 0, 20),
    'gamma': hp.uniform('gamma', 0, 20),
    'kernel': hp.choice('kernel', ['linear', 'sigmoid', 'rbf', 'poly']),
}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss': -acc, 'status':STATUS_OK}
trials = Trials()
best = fmin(fn=f, space=space_svm, algo=tpe.suggest, max_evals=100, trials=trials)
print('best', best)

parameters = ['C', 'gamma', 'kernel']
cols = len(parameters)
f, axes = plt.subplot(nrows=1, ncols=cols, figsize=(20,5))
cmap = plt.cm.jet
for i, val in enumerate(parameters):
    xs = np.array(t['misc']['vals'][val] for t in trials.trials).ravel()
    ys = [-t['result']['loss'] for t in trials.trials]
    axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25, c=cmap(float(i)/len(parameters)))
    axes[i].set_title(val)
    axes[i].set_ylim([0.9, 1.0])
#%% 决策树
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, fmin, STATUS_OK, Trials, hp
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
def hyperopt_train_test(params):
    clf = DecisionTreeClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space_dt = {'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features' : hp.choice('max_features', range(1,5)),
            'criterion': hp.choice('criterion',['gini','entropy']),}

def f(params):
    acc = hyperopt_train_test(params)
    return {'loss':-acc, 'status': STATUS_OK}
trial = Trials()
best = fmin(fn=f, space=space_dt, max_evals=100, trials=trial, algo=tpe.suggest)
#%% 随机森林
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, fmin, STATUS_OK, Trials, hp
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
iris = load_iris()
X, y = iris.data, iris.target
def hyperopt_train_test(params):
    clf = RandomForestClassifier(**params)
    return cross_val_score(clf, X, y).mean()

space_rf = {'max_depth': hp.choice('max_depth', range(1, 20)),
            'max_features' : hp.choice('max_features', range(1,5)),
            'n_estimators' : hp.choice('n_estimators', range(1,20)),
            'criterion': hp.choice('criterion',['gini','entropy']),}
best = 0
def f(params):
    global best
    acc = hyperopt_train_test(params)
    if acc > best:
        best = acc
    print('new best:', best, params)
    return {'loss':-acc, 'status': STATUS_OK}
trial = Trials()
best = fmin(fn=f, space=space_rf, max_evals=100, trials=trial, algo=tpe.suggest)
print('best:', best)
#%% 多模型调优
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from hyperopt import tpe, fmin, STATUS_OK, Trials, hp
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC

iris = load_iris()
X, y = iris.data, iris.target
def hyperopt_train_test(params):
    t = params['type']
    del params['type']
    if t == 'naive_bayes':
        clf = BernoulliNB(**params)
    elif t == 'svm':
        clf = SVC(**params)
    elif t == 'dtree':
        clf = DecisionTreeClassifier(**params)
    elif t == 'knn':
        clf = KNeighborsClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X, y).mean()

space = hp.choice('classfier_type', [{
                                        'type':'naive_bayes',
                                        'alpha': hp.uniform('alpha', 0.0, 2.0)},
                                    {'type' : 'svm',
                                     'gamma': hp.uniform('gamma', 0, 20.0),
                                     'kernel' : hp.choice('kernel',['linear', 'rbf']),
                                     'C' : hp.uniform('C', 0, 10.0)},
                                    {'type': 'dtree',
                                     'max_features' : hp.choice('max_features', range(1,5)),
                                     'max_depth': hp.choice('max_depth', range(1, 20)),
                                     'criterion': hp.choice('criterion', ['gini', 'entropy'])},
                                    {'type': 'knn',
                                     'n_neighbors': hp.choice('knn_n_neighbors', range(1, 50))}
                                    ])
count = 0
best = 0
def f(params):
    global best, count
    count += 1
    acc = hyperopt_train_test(params.copy())
    if acc > best:
        print('new best:', acc, 'using', params['type'])
        best = acc
    if count % 50 == 0:
        print('iters:', count, 'acc', acc, 'using', params)
    return {'loss':-acc, 'status': STATUS_OK}

trial = Trials()
best = fmin(fn=f, space=space, max_evals=100, trials=trial, algo=tpe.suggest)
print('best:', best)