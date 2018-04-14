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

test = test_import.drop('click_time')

#Modify training set 1. Dropping all duplicate os, channel, device combinations. 2. Adding the minority data back. 
train = train_import[train_import['is_attributed']==0].drop_duplicates(['os','channel','device']).append(train_import[train_import['is_attributed']==0])

sub['click_id'] = test['click_id']
test.drop('click_id', axis = 1, inplace = True)

y = train['is_attributed']
train.drop(['is_attributed', 'attributed_time'], axis=1, inplace=True)


params = {'eta': 0.05, 
          'max_depth': 6, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':100,
          'alpha':4,
          'objective': 'binary:logistic', 
          'eval_metric': 'auc', 
          'random_state': 99, 
          'silent': True}

x1, x2, y1, y2 = skl.train_test_split(train, y, test_size=0.1, random_state=55)

watchlist = [(xgb.DMatrix(x1, y1), 'train'), (xgb.DMatrix(x2, y2), 'valid')]

#Fit model
model = xgb.train(params, xgb.DMatrix(x1, y1), 250, watchlist, maximize=True, verbose_eval=10)
print("training complete")

sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)

          
