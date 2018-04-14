import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot
import xgboost as xgb
import sklearn.cross_validation as skl
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

#Import data
train_import = pd.read_csv("train_sample.csv")
test_import = pd.read_csv("test.csv")

def dataPreProcessTime(df):
    df['click_time'] = pd.to_datetime(df['click_time'])
    df['click_time'] = df['click_time'].apply(lambda x: x.strftime('%H')).astype(int)
        
    return df
    
    
test_import = dataPreProcessTime(test_import)
train_import = dataPreProcessTime(train_import)

# Make list of bad IP addresses
train_import.grouby('ip')['is_attributed'].


y_full = train_import['is_attributed']
train_full = train_import.drop(['is_attributed', 'attributed_time'], axis = 1)

#Modify training set 1. Dropping all duplicate os, channel, device combinations. 2. Adding the minority data back. 
train = train_import[train_import['is_attributed']==0].drop_duplicates(['os','channel','device']).append(train_import[train_import['is_attributed']==1])


sub = pd.DataFrame()
sub['click_id'] = test_import['click_id']
test_import.drop('click_id', axis = 1, inplace = True)

y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


params = {'eta': 0.08, 
          'max_depth': 5, 
          'booster' : 'gbtree',
          'gamma' : 0,
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':70,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}

x1, x2, y1, y2 = skl.train_test_split(train, y, test_size=0.1, random_state=55)

watchlist = [(xgb.DMatrix(train_full, y_full), 'train')]

#Fit model
model = xgb.train(params, xgb.DMatrix(train,y), 400, watchlist, maximize=True, verbose_eval=10)
print("training complete")

xgb.cv(params, xgb.DMatrix(train, y), 300, nfold = 5, seed=0)

sub['is_attributed'] = model.predict(xgb.DMatrix(test_import), ntree_limit=model.best_ntree_limit)

          
