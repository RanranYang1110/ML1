#-*- coding:utf-8 -*-
# @author: qianli
# @file: gbdt_test2.py
# @time: 2019/07/30
import xgboost as xgb
import numpy as np
import pandas as pd
import seaborn as sns
from hyperopt import hp
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')
import os
os.chdir(r"D:\7-学习资料\a-study\machinlearning\万门大学课件\课件（三）\0816\0816 下\集成模型实战")
dataset = pd.read_csv('bike.csv')
dataset.info()
#%%
'''
datetime - hourly date + timestamp
season -
  1 = spring
  2 = summer
  3 = fall
  4 = winter
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather -
    1: Clear, Few clouds, Partly cloudy, Partly cloudy
    2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals
'''

'''目标：一个回归问题，预测count, count = registered + casual
   评价函数：sqrt(mean(ln(pi+1) - ln(ai+1)))'''

'''特征处理'''
dataset['datetime'] = pd.to_datetime(dataset['datetime'])
dataset['day'] = dataset['datetime'].map(lambda x:x.day)

'''建模'''
def assing_test_samples(data, last_training_day=0.3, seed=1):
    days = data.day.unique()
    np.random.seed(seed)
    np.random.shuffle(days)
    test_days = days[: int(len(days) * last_training_day)]
    data['is_test'] = data.day.isin(test_days)
assing_test_samples(dataset)

def select_features(data):
    #只把数值特征取出来
    columns = data.columns[(data.dtypes == np.int64) | (data.dtypes == np.float64) | (data.dtypes == np.bool)].values
    return [feat for feat in columns if feat not in ['count', 'casual', 'registered'] and 'log' not in feat]

def get_X_y(data, target_variable):
    ## Wrapper
    features = select_features(data=data)
    X = data[features].values
    y = data[target_variable].values
    return X,y

def train_test_split(train, target_variable):
    ## 划分训练集和测试集
    df_train = train[train.is_test == False]
    df_test = train[train.is_test == True]
    X_train, y_train = get_X_y(df_train, target_variable)
    X_test, y_test = get_X_y(df_test, target_variable)
    return X_train, y_train, X_test, y_test

def fit_and_predict(train, model, target_variable):
    ## 模型拟合预测
    X_train, y_train, X_test, y_test = train_test_split(train, target_variable)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred

def post_pred(y_pred):
    ## 把预测为负的值全部置0
    y_pred[y_pred < 0] = 0
    return y_pred

def rmsle(y_true, y_pred, y_pred_only_positive=True):
    # 模型评价
    if y_pred_only_positive: y_pred = post_pred(y_pred)
    diff = np.log(y_pred + 1) - np.log(y_true + 1)
    mean_error = np.square(diff).mean()
    return np.sqrt(mean_error)

def count_prediction(train, model, target_variable='count'):
    y_test, y_pred = count_prediction(train, model, target_variable)
    return rmsle(y_true=y_test, y_pred=y_pred)

#Dummy Regressor 把所有instance全部预测成众数、平均数等
# print('dummy',count_prediction(dataset, DummyRegressor(strategy='mean')))
# print('xgboost', count_prediction(dataset, xgb.XGBRegressor()))

def registered_casual_prediction(train, model):
    _, registered_pred = fit_and_predict(train, model, 'registered')
    _, casual_pred = fit_and_predict(train, model, 'casual')
    y_test = train[train.is_test == True]['count']
    y_pred = registered_pred + casual_pred
    return rmsle(y_test, y_pred)

def log_registered_casual_prediction(train, model):
    _, registered_pred = fit_and_predict(train, model, 'registered_log')
    _, casual_pred = fit_and_predict(train, model, 'casual_log')
    y_test = train[train.is_test == True]['count']
    y_pred = registered_pred + casual_pred
    return rmsle(y_test, y_pred)

def importance_features(model, data):
    impdf = []
    fscore = model.get_booster().get_fscore()
    maps_name = dict([("f(0)".format(i), col) for i, col in enumerate(data.columns)])
    for ft, score in fscore.items():
        impdf.append({'feature': maps_name[ft], 'importance': score})
    impdf = pd.DataFrame(impdf)
    impdf = impdf.sort_values(by='importance', ascending=False).reset_index(drop=True)
    impdf['importance'] /= impdf['importance'].sum()
    impdf.index = impdf['feature']
    return impdf

def draw_importance_features(model, train):
    impdf = importance_features(model, train)
    return impdf.plot(kind='bar', title='Importance Features', figsize=(20,8))

def etl_datetime(df):
    df['year'] = df['datetime'].map(lambda x: x.year)
    df['month'] = df['datetime'].map(lambda x: x.month)
    df['hour'] = df['datetime'].map(lambda x: x.hour)
    df['minute'] = df['datetime'].map(lambda x: x.minute)
    df['dayofweek'] = df['datetime'].map(lambda x: x.dayofweek)
    df['weekend'] = df['datetime'].map(lambda x: x.dayofweek in [5,6])
etl_datetime(dataset)

for name in ['registered', 'casual']:
    dataset['{0}_log'.format(name)] = dataset[name].map(lambda x : np.log2(x+1))

def objective(space):
    model = xgb.XGBRegressor(
        max_depth = int(space['max_depth']),
        n_es_estimators = int(space['n_estimators']),
        subsample = space['subsample'],
        colsample_bytree = space['colsample_bytree'],
        learning_rate = space['learning_rate'],
        reg_alpha = space['reg_alpha']
    )
    X_train, y_train, X_test, y_test = train_test_split(dataset, 'count')
    eval_set = [(X_train, y_train), (X_test, y_test)]
    _, registered_pred = fit_and_predict(dataset, model, 'registered_log')
    _, casual_pred = fit_and_predict(dataset, model, 'casual_log')

    y_test = dataset[dataset.is_test == True]['count']
    y_pred = (np.exp2(registered_pred) - 1) + (np.exp2(casual_pred) - 1)
    score = rmsle(y_test, y_pred)
    print('SCORE', score)
    return {'loss': score, 'status': STATUS_OK}

space = {
    'max_depth' : hp.quniform('x_max_depth', 2, 20, 1),
    'n_estimators' : hp.quniform('n_estimators', 100, 1000, 1),
    'subsample' : hp.uniform('x_subsample', 0.8, 1),
    'colsample_bytree': hp.uniform('x_colsample_bytree', 0.1, 1),
    'learning_rate' : hp.uniform('x_learning_rate', 0.01, 0.1),
    'reg_alpha' : hp.uniform('x_reg_alpha', 0.1, 1)
}
trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=15, trials=trials)
print(best)
