#-*- coding:utf-8 -*-
# @author: qianli
# @file: parameterOptimization.py
# @time: 2019/08/01
'''GridsearchCV'''
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report

digits = datasets.load_digits()
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

#设置gridsearch的参数
tuned_parameters = [{'kernel': ['rbf'], 'gamma' : [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#设置模型评估的方法
scores = ['precision', 'recall']
for score in scores:
    print('#Tuning hyper-parameters for %s' % score)
    print()
    #构造这个GridSearch的分类器，5-fold
    clf = GridSearchCV(SVC(), tuned_parameters, cv=5, scoring='%s_macro' % score)
    # 只在训练集上面做k-fold,然后返回最优的模型参数
    clf.fit(X_train, y_train)
    print('Best parameters set found on development set:')
    print()
    #输出最优的模型参数
    print(clf.best_params_)
    print()
    print('Grid scores on development set:')
    print()
    # for params, mean_score, scores in clf.cv_results_:
    #     print('%0.3f(+/-%0.03f for %r' % (mean_score, scores.std() * 2, params))
    # print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print('%0.3f(+/-%0.03f) for %r' % (mean, std * 2, params))
    print('Detailed classification report:')
    print()
    print('The model is trained on the full development set.')
    print('The scores are computed on the full evaluation set.')
    print()

    #在测试集上测试最优的模型的泛化能力
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
#%%
'''有关用于同时评估多个指标的GridSearchCV示例'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_hastie_10_2
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
X, y = make_hastie_10_2(n_samples=8000, random_state=42)
scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
gs = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid={'min_samples_split': range(2, 403, 10)},
                                         scoring=scoring, cv=5, refit='AUC', return_train_score=True)
gs.fit(X, y)
results = gs.cv_results_
plt.figure(figsize=(13,13))
plt.title('GridSearchCV evaluating using multile scores simultaneously', fontsize=16)
plt.xlabel('min_samples_split')
plt.ylabel('Score')
ax = plt.gca()
ax.set_xlim(0, 402)
ax.set_ylim(0.73, 1)

# Get the regular numpy array from the MakedArray
X_axis = np.array(results['param_min_samples_split'].data, dtype=float)
for scorer, color in zip(sorted(scoring), ['g','k']):
    for sample, style in (('train','--'), ('test', '-')):
        sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
        sample_score_std = results['std_%s_%s' % (sample, scorer)]
        ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                        sample_score_mean + sample_score_std,
                        alpha=0.1 if sample=='test' else 0, color=color)
        ax.plot(X_axis, sample_score_mean, style, color=color,
                alpha=1 if sample == 'test' else 0.7,
                label = "%s (%s)" % (scorer, sample))
    best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
    best_score = results['mean_test_%s' % scorer][best_index]
    # plot a dotted vertical line at the best score for that scorer marked by x
    ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)
    ax.annotate('%0.2f'%best_score,
                (X_axis[best_index], best_score + 0.005))
    plt.legend(loc='best')
    plt.grid(False)
    plt.show()
#%%
'''随机参数优化算法'''
import numpy as np
from time import time
from scipy.stats import randint as sp_randint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
digits = load_digits()
X, y = digits.data, digits.target

clf = RandomForestClassifier(n_estimators=20)

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print('Model with rank: {0}'.format(i))
            print('Mean validation score:{0:.3f}(std: {1:.3f})'.format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters:{0}".format(results['params'][candidate]))
            print("")

param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1,11),
              "min_samples_split": sp_randint(2,11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
n_iter_search = 20
random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search,
                                   cv=5, iid=False)
start = time()
random_search.fit(X, y)
print("RandomizedSearchCV took %.2f seconds for %d candidates."
      " parameter_settings." % ((time() - start), n_iter_search))
report(random_search.cv_results_)

param_grid = {"max_depth": [3, None],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
grid_search = GridSearchCV(clf, param_grid=param_grid, cv=5, iid=False)
start = time()
grid_search.fit(X, y)
print("GridSearchCV took %.2f seconds for %d candidates."
       % ((time() - start), len(grid_search.cv_results_['params'])))
report(grid_search.cv_results_)