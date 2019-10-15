'''交叉验证'''
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
tree_reg = DecisionTreeRegressor()

scores = cross_val_score(tree_reg, x_train, y_train, scoring='nen_mean_squared_error', cv=10)
rmse_scores = np.sqrt(-scores)

'''模型保存'''
from sklearn.externals import joblib
joblib.dump(my_model,'my_model.pkl')
my_model_loaded = joblib.load('my_model_pkl')

'''网格搜索'''
from sklearn.model_selection import GridSearchCV
param_grid = [ #首先评估第一个dict中的n_estimator 和max_features的所有3*4=12种超参数值组合
    #接着尝试第二个dict种的2*3种超参数值组合，对每个模型进行5次训练
    {'n_estimators': [3, 10, 30], 'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4]}
]
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_predpared, housing_labels)

'''随机搜索（应用于超参数的搜索范围较大时）RandomizedSearchCV'''

'''指出每个属性的相对重要程度'''
features_importances = grid_search.best_estimator_.feature_importances_

