# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 13:37:33 2018

@author: Polestar User
"""

import pandas as pd

import numpy as np

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import MinMaxScaler
from sklearn import ensemble, tree, linear_model
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.utils import shuffle


import matplotlib.pyplot as plt


train = pd.read_csv("C:\\Users\\Polestar User\\Desktop\\housing\\train.csv") 
test = pd.read_csv("C:\\Users\\Polestar User\\Desktop\\housing\\test.csv")

data = train.append(test)

a=data.isna().sum()

numerical = data.select_dtypes(include = ['float64', 'int64'])
numerical.head()


############# Categorical features

category = data.select_dtypes(include = ['O'])
category.head()

num=numerical.isna().sum()
plt.style.use('ggplot')
num.T.plot(kind='bar')

cat=category.isna().sum()
plt.style.use('ggplot')
cat.T.plot(kind='bar')

### FILLING MISSING VALUES

mean_mva=data['MasVnrArea'].mean()
numerical['MasVnrArea']=numerical['MasVnrArea'].fillna(mean_mva)


mean_frontage=numerical['LotFrontage'].mean()
numerical['LotFrontage']=numerical['LotFrontage'].fillna(mean_frontage)

numerical['GarageYrBlt'] = numerical['GarageYrBlt'].fillna(0)


### Deleting columns having high number of missing values from categorical

del category['PoolQC']
del category['Fence']
del category['MiscFeature']
del category['Alley']
del category['FireplaceQu']


#### Filling missing values from categorical


for col in ('GarageType', 'GarageFinish', 'GarageQual','GarageCond'):
    category[col] = category[col].fillna('NoGRG')
    
    
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
    category[col] = category[col].fillna('NoBSMT')
    
category['MasVnrType'] = category['MasVnrType'].fillna(category['MasVnrType'].mode()[0])

category['MSZoning'] = category['MSZoning'].fillna(category['MSZoning'].mode()[0])

category['Electrical'] = category['Electrical'].fillna(category['Electrical'].mode()[0])

category['Exterior1st'] = category['Exterior1st'].fillna('VinylSd')

category['Exterior2nd'] = category['Exterior2nd'].fillna('VinylSd')

category['Functional'] = category['Functional'].fillna('Typ')

category['SaleType'] = category['SaleType'].fillna('WD')

category['Utilities'] = category['Utilities'].fillna('AllPub')

category['KitchenQual'] = category['KitchenQual'].fillna('Gd')


###### Feature Engineering

numerical['Age']=numerical['YrSold'] - numerical['YearBuilt']

numerical['Porch'] =  (numerical['OpenPorchSF']+numerical['EnclosedPorch']+numerical['ScreenPorch']+numerical['3SsnPorch'])/4

numerical['TotalBath']=numerical['FullBath']+numerical['HalfBath']

numerical['TotalSF'] = numerical['TotalBsmtSF'] + numerical['1stFlrSF'] + numerical['2ndFlrSF']

df1 = data['SalePrice'].groupby(data['Neighborhood']).mean()

df2 = data['GrLivArea'].groupby(data['Neighborhood']).mean()

train.columns.values

type(df1)

df3 = df1/df2

allDict = {'Blmngtn':136.469825,
'Blueste':98.743268,
'BrDale':91.405609,
'BrkSide':103.763006,
'ClearCr':118.877344,
'CollgCr':133.715484,
'Crawfor':117.561846,
'Edwards':95.683487,
'Gilbert':117.499892,
'IDOTRR':87.651248,
'MeadowV':93.089657,
'Mitchel':120.746121,
'NAmes':111.307215,
'NPkVill':113.882238,
'NWAmes':109.417853,
'NoRidge':133.644829,
'NridgHt':165.072651,
'OldTown':86.654779,
'SWISU':79.162888,
'Sawyer':112.574422,
'SawyerW':116.961639,
'Somerst':141.157845,
'StoneBr':165.239905,
'Timber':138.477089,
'Veenker':155.083845}


numerical['PricePSF'] = category['Neighborhood'].copy()


numerical["PricePSF"].replace(allDict, inplace=True)

numerical.to_csv("C:\\Users\\Polestar User\\Desktop\\1234.csv")

#####

Allfeatures = pd.concat([numerical,category],axis=1)


####

Allfeatures.drop(['Utilities', 'RoofMatl', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating', 'LowQualFinSF',
               'BsmtFullBath', 'BsmtHalfBath', 'Functional', 'GarageYrBlt', 'GarageArea', 'GarageCond', 'WoodDeckSF',
               'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal'],
              axis=1, inplace=True)


Allfeatures['TotalSF'] = Allfeatures['TotalSF'].fillna(800)
Allfeatures['GarageCars'] = Allfeatures['GarageCars'].fillna(2)
Allfeatures['TotalBsmtSF'] = Allfeatures['TotalBsmtSF'].fillna(800)

####



del Allfeatures['SalePrice']
del Allfeatures['Condition2']
del Allfeatures['Electrical']
del Allfeatures['Exterior1st']
del Allfeatures['Exterior2nd']
del Allfeatures['HouseStyle']
del Allfeatures['GarageQual']


a=Allfeatures.isna().sum()

numeric_features = [f for f in Allfeatures.columns if Allfeatures[f].dtype != object]

scaler = MinMaxScaler()
o=scaler.fit(Allfeatures[numeric_features])
scaled = scaler.transform(Allfeatures[numeric_features])

for i, col in enumerate(numeric_features):
       Allfeatures[col] = scaled[:,i]


train1 = Allfeatures[0:1460]
test1 = Allfeatures[1460:]


X = train1.copy()

y = train['SalePrice']


X = pd.get_dummies(X)

test1 = pd.get_dummies(test1)

set(X)-set(test1)

y = np.log(y)


#######################################

def get_score(prediction, lables):    
    print('R2: {}'.format(r2_score(prediction, lables)))
    print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))

# Shows scores for train and validation sets    
def train_test(estimator, x_trn, x_tst, y_trn, y_tst):
    prediction_train = estimator.predict(x_trn)
    # Printing estimator
    print(estimator)
    # Printing train scores
    get_score(prediction_train, y_trn)
    prediction_test = estimator.predict(x_tst)
    # Printing test scores
    print("Test")
    get_score(prediction_test, y_tst)



x_train_st, x_test_st, y_train_st, y_test_st = train_test_split(X,y, test_size=0.1, random_state=200)



regr = linear_model.LinearRegression()
regr.fit(X,y)

pred = regr.predict(test1)

predtest=np.exp(pred)


len(predtest)


sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = predtest
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission1.csv',index=False)


#######Second Model#########

from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
lasso = LassoCV(alphas=[0.001,.0001,.0015],cv=10,random_state = 1).fit(X,y)

lasso.score(X,y)

lasso.alpha_

lasso.coef_

predtest = lasso.predict(test1)
predtest=np.exp(predtest)

sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = predtest
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission3.csv',index=False)



####### Third Model ################

ENSTest = linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10], l1_ratio=[.01, .1, .5, .9, .99], cv=10,max_iter=5000).fit(X,y)

ENSTest.alpha_

predtest = ENSTest.predict(test1)
predtest=np.exp(predtest)

sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = predtest
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission4.csv',index=False)


###### Forth Model ################

GBest = ensemble.GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(X, y)

predtest = GBest.predict(test1)
predtest=np.exp(predtest)

sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = predtest
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission6.csv',index=False)

####### model 5 by stacking ###########

GB_model = GBest.fit(X, y)
ENST_model = ENSTest.fit(X, y)


Final_labels = (np.exp(GB_model.predict(test1)) + np.exp(ENST_model.predict(test1))) / 2

sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = Final_labels
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission7.csv',index=False)



######### model 6 ##################


import xgboost as xgb

regr = xgb.XGBRegressor(
                 colsample_bytree=0.2,
                 gamma=0.0,
                 learning_rate=0.01,
                 max_depth=4,
                 min_child_weight=1.5,
                 n_estimators=7200,                                                                  
                 reg_alpha=0.9,
                 reg_lambda=0.6,
                 subsample=0.2,
                 seed=42,
                 silent=1)

regr.fit(X, y)

# Run prediction on training set to get a rough idea of how well it does.
y_pred = regr.predict(test1)
y_pred=np.exp(y_pred)


sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = y_pred
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\submission7.csv',index=False)



####### model 7 stacking xg + Elastic + GBM

Final_labels = (np.exp(GB_model.predict(test1)) + np.exp(ENST_model.predict(test1)) + np.exp(regr.predict(test1))) / 3

sub1 = pd.DataFrame()
sub1['Id'] = test['Id']
sub1['SalePrice'] = Final_labels
sub1.to_csv('C:\\Users\\Polestar User\\Desktop\\finalsubmit\\stackxgelasticgbm.csv',index=False)

