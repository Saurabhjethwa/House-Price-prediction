# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 12:46:05 2018

@author: Polestar User
"""

import pandas as pd

import numpy as np

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


import matplotlib.pyplot as plt


data = pd.read_csv("C:\\Users\\Polestar User\\Desktop\\train.csv") 
data.head()

a=data.isna().sum()
a.to_csv('C:\\Users\\Polestar User\\Desktop\\NA.csv')

print(data)

data.info()
data.describe()

############## Numerical Features

numerical = data.select_dtypes(include = ['float64', 'int64'])
numerical.head()


############# Categorical features

category = data.select_dtypes(include = ['O'])
category.head()



########## Plotting NA of Numerical

num=numerical.isna().sum()
plt.style.use('ggplot')
num.T.plot(kind='bar')

## for MasVnrArea

mean_mva=data['MasVnrArea'].mean()
numerical['MasVnrArea']=numerical['MasVnrArea'].fillna(mean_mva)

## For GarageYrBlt

numerical['GarageYearDiff'] = numerical['GarageYrBlt']-numerical['YearBuilt']


numerical['GarageYearDiff'].hist(bins=100)

plt.scatter(numerical['GarageYearDiff'],data['SalePrice'])

sns.lmplot(x="GarageYearDiff", y="SalePrice", data=numerical)


del numerical['GarageYrBlt']

## For LotFontage

numerical['LotFrontage'].hist(bins=100)

num1 = numerical.copy()

mean_frontage=numerical['LotFrontage'].mean()
num1['LotFrontage']=numerical['LotFrontage'].fillna(mean_frontage)
num1['LotFrontage'].hist(bins=100)

plt.scatter(numerical['LotFrontage'],data['SalePrice'])

plt.scatter(num1['LotFrontage'],data['SalePrice'])


######## top 10 features correlated with SalePrice

top = numerical.corr()['SalePrice'][:-1] # -1 because the latest row is SalePrice
features_list = top[abs(top) > 0.4].sort_values(ascending=False)

####################################################

sns.distplot(data['SalePrice'], color='r', bins=100, hist_kws={'alpha': 0.4});

### CORRELATAION mATRIX

corr = numerical.corr()
corr.style.background_gradient()

#### HEATMAP

abc2 = numerical.corr()

plt.figure(figsize = (20,10))
sns.heatmap(abc2[(abc2 >= 0.5) | (abc2 <= -0.4)], 
        xticklabels=abc2.columns,
        yticklabels=abc2.columns,linewidths=0.1)


#### Numerical Feature Reduction

num_copy=numerical.copy()

del num_copy['BsmtUnfSF']
del num_copy['TotalBsmtSF']
#del num_copy['GarageYrBlt']
del num_copy['TotRmsAbvGrd']
del num_copy['GarageArea']
del num_copy['2ndFlrSF']
del num_copy['BsmtFullBath']
del num_copy['1stFlrSF']
del num_copy['BedroomAbvGr']
del num_copy['MasVnrArea']


########### removing numerical after checking scatter plot

del num_copy['Id']
del num_copy['LowQualFinSF']
del num_copy['BsmtFinSF2']
del num_copy['PoolArea']
del num_copy['MiscVal']
del num_copy['BsmtHalfBath']
del num_copy['Fireplaces']


category['MSSubClass'] = num_copy['MSSubClass']
category['MoSold'] = num_copy['MoSold']

category['MSSubClass'] = category['MSSubClass'].astype('category')
category['MoSold'] = category['MoSold'].astype('category')

del num_copy['MSSubClass']
del num_copy['MoSold']


######### Feature Engineering

## 1

num_copy['YearDiff']=num_copy['YrSold'] - num_copy['YearBuilt']

del num_copy['YrSold']
del num_copy['YearBuilt']


import numpy as np

x=num_copy['YearDiff']
y=num_copy['SalePrice']

plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"r--")

plt.show()


##

x=num_copy['YearRemodAdd']
y=num_copy['SalePrice']

plt.scatter(x, y)

z = np.polyfit(x, y, 1)
p = np.poly1d(z)
plt.plot(x,p(x),"b--")

plt.show()


del num_copy['YearRemodAdd']




## 2

del num_copy['3SsnPorch']

num_copy['Porch'] =  (num_copy['OpenPorchSF']+num_copy['EnclosedPorch']+num_copy['ScreenPorch'])/3

plt.scatter(num_copy['Porch'], num_copy['SalePrice'])


del num_copy['OpenPorchSF']
del num_copy['EnclosedPorch']
del num_copy['ScreenPorch']


## 3

num_copy['FullBath'] = data['FullBath']

num_copy['Bath']=num_copy['FullBath']+num_copy['HalfBath']

del num_copy['FullBath']
del num_copy['HalfBath']


len(num_copy.columns.values)



############ Plotting NA values of Categories

cat = category.isna().sum()
plt.style.use('ggplot')
cat.T.plot(kind='bar')

del category['PoolQC']
del category['Fence']
del category['MiscFeature']
del category['Alley']
del category['FireplaceQu']

##########################################



import statsmodels.api as sm # import statsmodels 


import statsmodels.formula.api as smf

mod = smf.ols(formula='SalePrice ~ LotArea + OverallQual + OverallCond + BsmtFinSF1 + GrLivArea + KitchenAbvGr + GarageCars + WoodDeckSF + YearDiff + Porch + Bath', data=num_copy)

res = mod.fit()

pred = res.predict()

print(res.summary())

true = numerical['SalePrice']


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mean_absolute_percentage_error(true,pred)


del num_copy['GarageYearDiff']
del num_copy['LotFrontage']

num_copy.columns.values

num_copy['SalePrice'] = numerical['SalePrice']


###################################################################################################################################################

############ Plotting NA values of Categories

cat = category.isna().sum()
plt.style.use('ggplot')
cat.T.plot(kind='bar')

del category['PoolQC']
del category['Fence']
del category['MiscFeature']
del category['Alley']
del category['FireplaceQu']

len(category.columns.values)


########### Category

fig, axes = plt.subplots(round(len(category.columns) / 4), 4, figsize=(12, 30))
for i, ax in enumerate(fig.axes):
    if i < len(category.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        sns.countplot(x=category.columns[i], alpha=0.7, data=category, ax=ax)
fig.tight_layout()

## 1. Street

sns.set_style("whitegrid")
ax = sns.boxplot(x="Street", y="SalePrice",data=data, palette="Set3")


## 2. Utilities

sns.set_style("whitegrid")
ax = sns.boxplot(x="Utilities", y="SalePrice",data=data, palette="Set3")

## 3. CentralAir

sns.set_style("whitegrid")
ax = sns.boxplot(x="CentralAir", y="SalePrice",data=data, palette="Set3")


## 4. Heating

ax = sns.boxplot(x="Heating", y="SalePrice",data=data, palette="Set3")

## 5. 

ax = sns.boxplot(x="GarageQual", y="SalePrice",data=data, palette="Set3")

ax = sns.boxplot(x="GarageCond", y="SalePrice",data=data, palette="Set3")

pd.crosstab(data['GarageCond'],data['GarageQual'])

#del category['GarageQual']

category['GarageOverall'] = np.where((category['GarageCond']=='TA') | (category['GarageCond']=='Gd'), 'AboveAvg', 'BelowAvg')

ax = sns.boxplot(x="GarageOverall", y="SalePrice",data=category, palette="Set3")

print(sns.countplot(x=category['GarageOverall'], alpha=0.7, data=category))

category['SalePrice']=data['SalePrice']
#del category['GarageCond']

## 6.

ax = sns.boxplot(x="LandContour", y="SalePrice",data=category, palette="Set3")
ax = sns.boxplot(x="LandOverall", y="SalePrice",data=category, palette="Set3")


pd.crosstab(data['LandContour'],data['LandSlope'])

category['LandOverall'] = np.where((category['LandContour']=='Lvl') & (category['LandSlope']=='Gtl'), 'Flat', 'Other')

#del category['LandContour']
#del category['LandSlope']



