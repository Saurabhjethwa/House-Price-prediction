# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 11:40:08 2018

@author: Polestar User
"""

import pandas as pd

import numpy as np

import seaborn as sns

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


############ Plotting NA values of Categories

cat = category.isna().sum()
plt.style.use('ggplot')
cat.T.plot(kind='bar')

del category['PoolQC']
del category['Fence']
del category['MiscFeature']
del category['Alley']
del category['FireplaceQu']

############ Category with SalePrice

SP = data['SalePrice']
category.iloc[0,0] = SP 
category.head()


########## Plotting NA of Numerical

num=numerical.isna().sum()
plt.style.use('ggplot')
num.T.plot(kind='bar')


############# Histogram for Numerical Data and distribution

numerical.hist(figsize=(16, 20), bins=100, xlabelsize=10, ylabelsize=10);

sns.distplot(data['SalePrice'], color='r', bins=100, hist_kws={'alpha': 0.4});

################ Categorical frequency table

fig, axes = plt.subplots(round(len(category.columns) / 4), 4, figsize=(12, 30))
for i, ax in enumerate(fig.axes):
    if i < len(category.columns):
        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=90)
        sns.countplot(x=category.columns[i], alpha=0.7, data=category, ax=ax)
fig.tight_layout()


################# Correlation heatmap

abc2 = num_copy.corr()

plt.figure(figsize = (20,10))
sns.heatmap(abc2[(abc2 >= 0.5) | (abc2 <= -0.4)], 
        xticklabels=abc2.columns,
        yticklabels=abc2.columns,linewidths=0.1)

################ Scatter Plot

sns.pairplot(data=numerical,x_vars=numerical.columns,y_vars=['SalePrice'])

############### Top Correlated variables with SalePrice

top_corr = numerical.corr()['SalePrice'][:-1] 
top_corr_list = top_corr[abs(top_corr) > 0.5]

############# removing numerical after checking heatmap

num_copy=numerical.copy()

del num_copy['BsmtUnfSF']
del num_copy['TotalBsmtSF']
del num_copy['GarageYrBlt']
del num_copy['TotRmsAbvGrd']
del num_copy['GarageArea']
del num_copy['2ndFlrSF']
del num_copy['BsmtFullBath']
#del num_copy['FullBath']
del num_copy['1stFlrSF']
del num_copy['BedroomAbvGr']



len(num_copy.columns.values)


########### removing numerical after checking scatter plot

del num_copy['Id']
del num_copy['LowQualFinSF']
del num_copy['BsmtFinSF2']
del num_copy['PoolArea']
del num_copy['MiscVal']
del num_copy['YrSold']
del num_copy['BsmtHalfBath']
del num_copy['Fireplaces']


category['MSSubClass'] = num_copy['MSSubClass']
category['OverallQual'] = num_copy['OverallQual']
category['OverallCond'] = num_copy['OverallCond']
category['MoSold'] = num_copy['MoSold']




category['CentralAir']


del num_copy['MSSubClass']
del num_copy['OverallQual']
del num_copy['OverallCond']
del num_copy['MoSold']

len(category.columns.values)
category= category.astype('category')



############## Categorical Feature Reduction

cat_copy=category.copy()

del cat_copy['Street']
del cat_copy['Condition2']
del cat_copy['Utilities']
del cat_copy['RoofMatl']


cat_copy['MoSold'] = cat_copy['MoSold'].astype('category')

category['OverallQual'] = category['OverallQual'].astype('category')
category['MSSubClass'] = category['MSSubClass'].astype('category')
category['OverallCond'] = category['OverallCond'].astype('category')
category['MoSold'] = category['MoSold'].astype('category')



############ feature engineering



num_copy['YearSold']=data['YrSold']
num_copy['YearDiff']=num_copy['YearSold'] - num_copy['YearBuilt']

del num_copy['YearSold']
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


###################################################

len(data.columns.values)

del num_copy['3SsnPorch']

num_copy['Porch'] =  (num_copy['OpenPorchSF']+num_copy['EnclosedPorch']+num_copy['ScreenPorch'])/3

plt.scatter(num_copy['Porch'], num_copy['SalePrice'])


del num_copy['OpenPorchSF']
del num_copy['EnclosedPorch']
del num_copy['ScreenPorch']
###########################

num_copy['FullBath'] = data['FullBath']

num_copy['Bath']=num_copy['FullBath']+num_copy['HalfBath']

del num_copy['FullBath']
del num_copy['HalfBath']

#############################

#removing MasVnrArea because distribution between woodeck and masnvarea is same 

del num_copy['MasVnrArea']


##########################
  
plt.scatter(data['GrLivArea'],data['SalePrice'])


    
num1 = data['Neighborhood'].value_counts()


data['Neighborhood'].hist(bins=100)

plt.plot(data['PoolQC'],data['SalePrice'])



plt.plot(numerical['LotArea'],'o')



numerical['BsmtUnfSF'].corr(numerical['SalePrice'])


abc=numerical.iloc[:,0:10]


pd.crosstab(data['SaleType'],data['SalePrice'])





corr = corr(numerical,numerical['SalePrice'])
corr.style.background_gradient()



######### Insights ############

#1
sns.pointplot(x = "SaleType", y = "SalePrice",hue="SaleType",data = data);
plt.show()


sns.boxplot(x='SaleCondition', y='SalePrice', data=data)

sns.lmplot(x='YearRemodAdd', y='SalePrice', data=data, size=8)





