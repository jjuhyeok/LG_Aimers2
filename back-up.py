import pandas as pd
import random
import os
import numpy as np
from collections import Counter
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
import optuna
from sklearn.model_selection import StratifiedKFold
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import f1_score
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
seed_everything(37) # Seed 고정
train = pd.read_csv('./DATA/train.csv')
test = pd.read_csv('./DATA/test.csv')



def log(train,test):
    col_list = train.columns
    for col in col_list :
        if '제곱' in col:
            train[col+'log'] = np.log1p(train[col])
            test[col+'log'] = np.log1p(test[col])
    return train,test

def del_columns(train, test):
    col_list = train.columns
    nan_list = []
    nan_cnt = []
    nan_col = []
    full_list = []
    for col in col_list:
        if train[col].isnull().sum() == 0 :
            full_list.append(col)
            continue
        nan_list.append([col, train[col].isnull().sum()])
        nan_cnt.append(train[col].isnull().sum())
        nan_col.append(col)

    '''모든값이 결측값이면 제거'''
    del_col = []
    for col in nan_list :
        if col[1] == len(train) :
            del_col.append(col[0])
    train = train.drop(columns=del_col)
    test = test.drop(columns=del_col)

    '''값이 1개 존재하면 제거'''
    del_col = []
    col_list = train.describe().columns
    for col in col_list :
        if col == 'Y_Class':
            continue
        if col == 'Y_Quality':
            continue
        if col == 'LINE':
            continue
        if col == 'PRODUCT_CODE':
            continue
        if train[col].nunique()==1 :
            del_col.append(col)
    train = train.drop(columns=del_col)
    test = test.drop(columns=del_col)
    
    return train,test
   
def make_train_test_dataset(train,test):
    train_x = train.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE','Y_Class','Y_Quality'])
    test_x = test.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])
    train_y = train['Y_Quality']
    train_w = train[['Y_Class']]
    return train_x, test_x, train_y, train_w

def fillna(train,test,value):
    train = train.fillna(value)
    test = test.fillna(value)
    return train,test

def labelencoder(train,test,col_list):
    qual_col = col_list
    for i in qual_col:
        le = LabelEncoder()
        le = le.fit(train[i])
        train[i] = le.transform(train[i])

        for label in np.unique(test[i]): 
            if label not in le.classes_: 
                le.classes_ = np.append(le.classes_, label)
        test[i] = le.transform(test[i]) 
    return train,test

def multi_col(train,test) : 
    col_list = train.columns
    for col in col_list :
        if col == 'LINE':
            continue
        train[col+'제곱'] = (train[col]**2)//100
        test[col+'제곱'] = (test[col]**2)//100
    return train,test




# y quality scaling
train['Y_Quality'] = train['Y_Quality'].map(lambda x : np.log(x))

train, test = del_columns(train,test)

trainA_31 = train[train['PRODUCT_CODE']=='A_31']


testA_31 = test[test['PRODUCT_CODE']=='A_31']


trainA_31, testA_31 = del_columns(trainA_31,testA_31)


trainA_31_x,testA_31_x, trainA_31_y, trainA_31_w = make_train_test_dataset(trainA_31,testA_31)

###
trainA_31_fe = pd.concat([trainA_31_x, trainA_31_y],axis=1)
trainA_31_fe = pd.concat([trainA_31_fe, trainA_31_w],axis=1)
testA_31_x_fe = testA_31_x

trainA_31_1 = trainA_31_fe[(trainA_31_fe['LINE'] == 'T050304') | (trainA_31_fe['LINE'] == 'T010305') | (trainA_31_fe['LINE'] == 'T010306')]
trainA_31_2 = trainA_31_fe[(trainA_31_fe['LINE'] == 'T050307')]
testA_31_x_1 = testA_31_x_fe[(testA_31_x_fe['LINE'] == 'T050304') | (testA_31_x_fe['LINE'] == 'T010305') | (testA_31_x_fe['LINE'] == 'T010306')]
testA_31_x_2 = testA_31_x_fe[(testA_31_x_fe['LINE'] == 'T050307')]

trainA_31_x_1 = trainA_31_1.drop(['Y_Quality'],axis=1)
trainA_31_y_1 = trainA_31_1['Y_Quality']
trainA_31_x_2 = trainA_31_2.drop(['Y_Quality'],axis=1)
trainA_31_y_2 = trainA_31_2['Y_Quality']




trainA_31_x_1,testA_31_x_1 = labelencoder(trainA_31_x_1,testA_31_x_1,['LINE'])
trainA_31_x_2,testA_31_x_2 = labelencoder(trainA_31_x_2,testA_31_x_2,['LINE'])

trainA_31_x_1,testA_31_x_1 = fillna(trainA_31_x_1,testA_31_x_1,-1)
trainA_31_x_2,testA_31_x_2 = fillna(trainA_31_x_2,testA_31_x_2,-1)

print(trainA_31_x_2.columns)



from catboost import *


kfold = StratifiedKFold(n_splits=10, shuffle=True)


copy_1 = trainA_31_x_1['Y_Class']
copy_2 = trainA_31_x_2['Y_Class']
trainA_31_x_1.drop(['Y_Class'],axis=1,inplace=True)
trainA_31_x_2.drop(['Y_Class'],axis=1,inplace=True)
print(len(trainA_31_x_1))
print(len(trainA_31_y_1))

print(len(trainA_31_x_2))
print(len(trainA_31_y_2))

def objective(trial):
    params = {
            'iterations':trial.suggest_int("iterations", 300, 1000),
            'learning_rate' : trial.suggest_uniform('learning_rate',0.1, 1),
            'depth': trial.suggest_int('depth',5, 16),
            'min_data_in_leaf': trial.suggest_int('min_data_in_leaf',1,30),
            'reg_lambda': trial.suggest_uniform('reg_lambda',30,100),
            'subsample': trial.suggest_uniform('subsample',0.3,1),
            'random_strength': trial.suggest_uniform('random_strength',10,100),
            'od_wait':trial.suggest_int('od_wait', 10, 150),
            'leaf_estimation_iterations': trial.suggest_int('leaf_estimation_iterations',1,20),
            'bagging_temperature' :trial.suggest_loguniform('bagging_temperature', 1, 100),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0., 1.0),
            'random_state' : 9555,
            'verbose' : 0,
        }
    #'task_type' : 'GPU',
    #"eval_metric":'RMSE',

    valid_class = copy_2
    valid_class = pd.DataFrame(valid_class,columns = ['Y_Class'])
    valid_class = valid_class.reset_index(drop = True)
    '''
    trainA_31_x_1.drop(['Y_Class'],axis=1,inplace=True)
    trainA_31_x_2.drop(['Y_Class'],axis=1,inplace=True)
    '''
    cat = CatBoostRegressor(**params)
    cat.fit(trainA_31_x_1, trainA_31_y_1, 
              verbose=False)
    cat_pred = cat.predict(trainA_31_x_2)
    
    y_valid = cat_pred
    y_valid = pd.DataFrame(y_valid,columns = ['Y_Quality'])
    y_valid = y_valid.reset_index(drop = True)
    y_valid['Y_Class2'] = 1
    y_valid.loc[(y_valid['Y_Quality']<-0.64421190232267), 'Y_Class2'] = 0
    y_valid.loc[(y_valid['Y_Quality']>-0.6256814053066195), 'Y_Class2'] = 2
    score = f1_score(valid_class, y_valid['Y_Class2'], average = 'macro')
    print('Score = %lf'%score)
    return score

study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=550, show_progress_bar=True)

#study.optimize(lambda trial: objective_xgb(trial, train_x, train_y), n_trials=100)
print('Best trial: score {},\nparams {}'.format(study.best_trial.value, study.best_trial.params))


param = study.best_trial.params



'''
xgbc = XGBClassifier(**param, tree_method='gpu_hist', gpu_id=0,random_state=37)
xgbc.fit(train_x, train_y)

pred = xgbc.predict(test_x)
submit = pd.read_csv('./sample_submission.csv')
submit['Y_Class'] = pred
submit.to_csv('./submit.csv', index=False)
'''
