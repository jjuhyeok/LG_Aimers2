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
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
'''
train_x = train_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP', 'Y_Class', 'Y_Quality'])
train_y = train_df['Y_Class']

test_x = test_df.drop(columns=['PRODUCT_ID', 'TIMESTAMP'])
train_x = train_x.fillna(0) # NaN 0으로 채우기
test_x = test_x.fillna(0)
qual_col = ['LINE', 'PRODUCT_CODE']

for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_x[i])
    train_x[i] = le.transform(train_x[i])
    # test_x 데이터에만 존재하는 새로 출현한 데이터를 신규 클래스로 추가한다 (중요!!!)
    for label in np.unique(test_x[i]):
        if label not in le.classes_: # unseen label 데이터인 경우( )
            le.classes_ = np.append(le.classes_,label) # 미처리 시 ValueError발생
    test_x[i] = le.transform(test_x[i])
'''
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
    if col[1] == 598 :
        del_col.append(col[0])
train = train.drop(columns=del_col)
test = test.drop(columns=del_col)
train.head(3)



trainA_31 = train[train['PRODUCT_CODE']=='A_31']
train_T_31 = train[train['PRODUCT_CODE']=='T_31']
train_O_31 = train[train['PRODUCT_CODE']=='O_31']

testA_31 = test[test['PRODUCT_CODE']=='A_31']
test_T_31 = test[test['PRODUCT_CODE']=='T_31']
test_O_31 = test[test['PRODUCT_CODE']=='O_31']



col_list = train.columns
nan_listA_31 = []
nan_cntA_31 = []
nan_colA_31 = []
full_listA_31 = []
for col in col_list:
    if trainA_31[col].isnull().sum() == 0 :
        full_listA_31.append(col)
        continue
    nan_listA_31.append([col, trainA_31[col].isnull().sum()])
    nan_cntA_31.append(trainA_31[col].isnull().sum())
    nan_colA_31.append(col)
    

del_col = []
for col in nan_listA_31 :
    if col[1] == len(trainA_31) :
        del_col.append(col[0])
trainA_31 = trainA_31.drop(columns=del_col)
testA_31 = testA_31.drop(columns=del_col)


del_col = []
col_list = trainA_31.columns
for col in col_list[6:] :
    if trainA_31[col].nunique()==1 :
        del_col.append(col)
trainA_31 = trainA_31.drop(columns=del_col)
testA_31 = testA_31.drop(columns=del_col)


col_list = train.columns
nan_listO = []
nan_cntO = []
nan_colO = []
full_listO = []
for col in col_list:
    if train_O_31[col].isnull().sum() == 0 :
        full_listO.append(col)
        continue
    nan_listO.append([col, train_O_31[col].isnull().sum()])
    nan_cntO.append(train_O_31[col].isnull().sum())
    nan_colO.append(col)
    

del_col = []
for col in nan_listO :
    if col[1] == len(train_O_31) :
        del_col.append(col[0])
train_O_31 = train_O_31.drop(columns=del_col)
test_O_31 = test_O_31.drop(columns=del_col)


del_col = []
col_list = train_O_31.columns
for col in col_list[6:] :
    if train_O_31[col].nunique()==1 :
        del_col.append(col)
train_O_31 = train_O_31.drop(columns=del_col)
test_O_31 = test_O_31.drop(columns=del_col)


col_list = train.columns
nan_listT = []
nan_cntT = []
nan_colT = []
full_listT = []
for col in col_list:
    if train_T_31[col].isnull().sum() == 0 :
        full_listT.append(col)
        continue
    nan_listT.append([col, train_T_31[col].isnull().sum()])
    nan_cntT.append(train_T_31[col].isnull().sum())
    nan_colT.append(col)
    

del_col = []
for col in nan_listT :
    if col[1] == len(train_T_31) :
        del_col.append(col[0])
train_T_31 = train_T_31.drop(columns=del_col)
test_T_31 = test_T_31.drop(columns=del_col)


del_col = []
col_list = train_T_31.columns
for col in col_list[6:] :
    if train_T_31[col].nunique()==1 :
        del_col.append(col)
train_T_31 = train_T_31.drop(columns=del_col)
test_T_31 = test_T_31.drop(columns=del_col)

'''
trainA_31_x = trainA_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE','Y_Class','Y_Quality'])
testA_31_x = testA_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])
train_T_31_x = train_T_31.drop(columns=['PRODUCT_ID','TIMESTAMP','Y_Class','Y_Quality','PRODUCT_CODE'])
test_T_31_x = test_T_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])
train_O_31_x = train_O_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE','Y_Class','Y_Quality'])
test_O_31_x = test_O_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])

'''
#클래스 살리기
trainA_31_x = trainA_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE','Y_Quality'])
testA_31_x = testA_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])
train_T_31_x = train_T_31.drop(columns=['PRODUCT_ID','TIMESTAMP','Y_Quality','PRODUCT_CODE'])
test_T_31_x = test_T_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])
train_O_31_x = train_O_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE','Y_Quality'])
test_O_31_x = test_O_31.drop(columns=['PRODUCT_ID','TIMESTAMP','PRODUCT_CODE'])


# classification
trainA_31_y_c = trainA_31['Y_Class']
train_T_31_y_c = train_T_31['Y_Class']
train_O_31_y_c = train_O_31['Y_Class']

# regression
trainA_31_y_r = trainA_31['Y_Quality']
train_T_31_y_r = train_T_31['Y_Quality']
train_O_31_y_r = train_O_31['Y_Quality']

train_T_31_y_r = pd.DataFrame(train_T_31_y_r,columns = ['Y_Quality'])
train_T_31_y_r = train_T_31_y_r.reset_index(drop = True)
test_T = train_T_31_y_r

trainA_31_x=trainA_31_x.fillna(-1)
testA_31_x=testA_31_x.fillna(-1)
train_T_31_x=train_T_31_x.fillna(-1)
test_T_31_x=test_T_31_x.fillna(-1)
train_O_31_x=train_O_31_x.fillna(-1)
test_O_31_x=test_O_31_x.fillna(-1)

train_T_31_x['level0'] = 0
train_T_31_x['level1'] = 0
train_T_31_x['level0'][train_T_31_x['LINE'] == 'T100304'] = 1
train_T_31_x['level1'][train_T_31_x['LINE'] == 'T100306'] = 1

test_T_31_x['level0'] = 0
test_T_31_x['level1'] = 0
test_T_31_x['level0'][test_T_31_x['LINE'] == 'T100304'] = 1
test_T_31_x['level1'][test_T_31_x['LINE'] == 'T100306'] = 1

# qualitative to quantitative
qual_col = ['LINE']
for i in qual_col:
    le = LabelEncoder()
    le = le.fit(trainA_31_x[i])
    trainA_31_x[i] = le.transform(trainA_31_x[i])
    
    for label in np.unique(testA_31_x[i]): 
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    testA_31_x[i] = le.transform(testA_31_x[i]) 
print('Done.')


# qualitative to quantitative
qual_col = ['LINE']
for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_T_31_x[i])
    train_T_31_x[i] = le.transform(train_T_31_x[i])
    
    for label in np.unique(test_T_31_x[i]): 
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test_T_31_x[i] = le.transform(test_T_31_x[i]) 
print('Done.')


# qualitative to quantitative
qual_col = ['LINE']
for i in qual_col:
    le = LabelEncoder()
    le = le.fit(train_O_31_x[i])
    train_O_31_x[i] = le.transform(train_O_31_x[i])
    
    for label in np.unique(test_O_31_x[i]): 
        if label not in le.classes_: 
            le.classes_ = np.append(le.classes_, label)
    test_O_31_x[i] = le.transform(test_O_31_x[i]) 
print('Done.')



from catboost import *


kfold = StratifiedKFold(n_splits=10, shuffle=True)


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
    x_train, x_valid, y_train, y_valid = train_test_split(train_T_31_x, train_T_31_y_r, test_size=0.3, stratify=train_T_31_x['Y_Class'])

    valid_class = x_valid['Y_Class']
    valid_class = pd.DataFrame(valid_class,columns = ['Y_Class'])
    valid_class = valid_class.reset_index(drop = True)
    
    x_train.drop(['Y_Class'],axis=1,inplace=True)
    x_valid.drop(['Y_Class'],axis=1,inplace=True)
    cat = CatBoostRegressor(**params)
    cat.fit(x_train, y_train, eval_set=[(x_train,y_train),(x_valid,y_valid)],
              verbose=False)
    cat_pred = cat.predict(x_valid)
    
    y_valid = cat_pred
    y_valid = pd.DataFrame(y_valid,columns = ['Y_Quality'])
    y_valid = y_valid.reset_index(drop = True)
    y_valid['Y_Class2'] = 1
    y_valid.loc[(y_valid['Y_Quality']<0.52507), 'Y_Class2'] = 0
    y_valid.loc[(y_valid['Y_Quality']>0.5349), 'Y_Class2'] = 2
    print(y_valid)
    print(valid_class)
    score = f1_score(valid_class, y_valid['Y_Class2'], average = 'macro')
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
