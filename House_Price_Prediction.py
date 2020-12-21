# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # House Price Prediction

# ## 1. Environment Setup

# +
# To get multiple outputs in the same cell

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# +
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# +
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline

# +
# Set the required global options

# To display all the columns in dataframe
pd.set_option( "display.max_columns", None)
pd.set_option( "display.max_rows", None)
# -



# ## 2. Reading the Input data (csv) file

 house = pd.read_csv('./train.csv')

house.head()

# ## 3. Data Analysis & Cleaning

# Checking rows and columns - shape 
house.shape

# Getting the overview of Data types and Non-Null info
house.info()

# ### Handling Missing Values

# +
# Checking for any Null columns
house.isnull().sum().any()

house.shape[0]

# Finding the columns with more than 40% NULLs.
ser = house.isnull().sum()/len(house)*100
null_drps = ser[ser > 40]
null_drps

# +
# Dropping variables with more than 95% NULLs.
# Here, out of these 5, four of them has more than 80% NULLs.

house.drop(null_drps.index, axis='columns', inplace=True)

# Verifying, whether variables are successfully dropped
ser = house.isnull().sum()/len(house)*100
nulls = ser[ser > 0]
nulls
# -

# Checking the info of the remaining columns with NULLs
house[nulls.index].info()

# #### Imputation of Numerical variables

# Imputing Numerical variables
num_var = ['LotFrontage','MasVnrArea','GarageYrBlt']
house[num_var].describe()

# Plotting boxplot to understand outliers
plt.figure(figsize=(15,7))
for i,j in enumerate(num_var):
    plt.subplot(1,3,i+1)
    sns.boxplot(data=house, x=j)
plt.show();

# +
# There are outliers in LotFrontage and MasVnrArea.
# I would impute these with median as mean is eaffected by outliers
house['LotFrontage'] = house['LotFrontage'].fillna(house['LotFrontage'].median())
house['MasVnrArea'] = house['MasVnrArea'].fillna(house['MasVnrArea'].median())

# There are no outliers in GarageYrBlt. So, I would impute this with mean
house['GarageYrBlt'] = house['GarageYrBlt'].fillna(house['GarageYrBlt'].mean())
# -

# #### Imputation of Categorical variables

# Checking the count of each category
house['MasVnrType'].value_counts()

# Replacing it with it's mode i.e. None
house['MasVnrType'] = house['MasVnrType'].replace(np.nan, house["MasVnrType"].mode()[0])

# Checking the count of each category
house['BsmtQual'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a basement.
house['BsmtQual'].fillna(house["MasVnrType"].mode()[0], inplace=True)

# Checking the count of each category
house['BsmtCond'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a basement.
# house['BsmtCond'] = house['BsmtCond'].replace(np.nan, 'NA')
house['BsmtCond'].fillna(house["BsmtCond"].mode()[0], inplace=True)

# Checking the count of each category
house['BsmtExposure'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a basement.
# house['BsmtExposure'] = house['BsmtExposure'].replace(np.nan, 'NA')
house['BsmtExposure'].fillna(house["BsmtExposure"].mode()[0], inplace=True)

# Checking the count of each category
house['BsmtFinType1'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a basement.
# house['BsmtFinType1'] = house['BsmtFinType1'].replace(np.nan, 'NA')
house['BsmtFinType1'].fillna(house["BsmtFinType1"].mode()[0], inplace=True)

# Checking the count of each category
house['BsmtFinType2'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a basement.
# house['BsmtFinType2'] = house['BsmtFinType2'].replace(np.nan, 'NA')
house['BsmtFinType2'].fillna(house["BsmtFinType2"].mode()[0], inplace=True)

# Checking the count of each category
house['Electrical'].value_counts()

# Replacing it with it's mode i.e. SBrkr
# house['Electrical'] = house['Electrical'].replace(np.nan, 'NA')
house['Electrical'].fillna(house["Electrical"].mode()[0], inplace=True)

# Checking the count of each category
house['GarageType'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a Garage.
# house['GarageType'] = house['GarageType'].replace(np.nan, 'NA')
house['GarageType'].fillna(house["GarageType"].mode()[0], inplace=True)

# Checking the count of each category
house['GarageFinish'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a Garage.
# house['GarageFinish'] = house['GarageFinish'].replace(np.nan, 'NA')
house['GarageFinish'].fillna(house["GarageFinish"].mode()[0], inplace=True)

# Checking the count of each category
house['GarageQual'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a Garage.
# house['GarageQual'] = house['GarageQual'].replace(np.nan, 'NA')
house['GarageQual'].fillna(house["GarageQual"].mode()[0], inplace=True)

# Checking the count of each category
house['GarageCond'].value_counts()

# Replacing NaN values to NA which indicates that the property doesnt have a Garage.
# house['GarageCond'] = house['GarageCond'].replace(np.nan, 'NA')
house['GarageCond'].fillna(house["GarageCond"].mode()[0], inplace=True)

# Checking for any Null columns
house.isnull().sum().any()

# Dropping variable 'Id' as it has a monotonic increasing value, which would not add any value
house.drop(columns='Id', inplace=True)

# ### Deriving features from date

# +
# import datetime as dt
# present_yr = int(dt.datetime.now().year())
# -

# Deriving Original Age of the house at Point of Sale 
house['HouseOrigAgeAtPOS'] = house['YrSold'] - house['YearBuilt']

house['HouseReModAgeAtPOS'] = house['YrSold'] - house['YearRemodAdd']

house['GarageAgeAtPOS'] = house['YrSold'] - house['GarageYrBlt']

# Deriving a feature to store 1, if house is remodelled, otherwise 0
house['IsReMod'] = np.where(house['YearBuilt'] == house['YearRemodAdd'], 0,1)

house[['YearBuilt','YearRemodAdd','YrSold','HouseOrigAgeAtPOS','HouseReModAgeAtPOS','IsReMod','GarageAgeAtPOS','SalePrice']].head()

# Now, since the features are derived from the date variables, we can drop them.
house.drop(columns=['YearBuilt','YearRemodAdd','GarageYrBlt','YrSold'],inplace=True)

# +
# Dropping MoSold, SaleType and SaleCondition - 
# These variables won't be available at the time of new Sale n hence cannot be considered for Price Prediction

# house.drop(columns=['MoSold','SaleType','SaleCondition'],inplace=True)

house.drop(columns=['MoSold'],inplace=True)

# +
# MSSubClass, OverallQual and OverallCond store num values but are categorical informations.
# Thus, converting them to categories.

to_cat_vars = ['MSSubClass','OverallQual','OverallCond']
for i in to_cat_vars:
    house[i] = house[i].astype(object)
    
# Verifying the type conversion
house[to_cat_vars].info()
# -



# ### Data Exploration

# +
# house.describe()

# +
# house.sample(5)
# -

plt.figure(figsize=(18,13));
sns.heatmap(house.corr(), annot = False);

# - Early findings from the heatmap suggests that Sale Price of house is faily correlated with 
#     - HouseOrigAgeAtPOS
#     - HouseReModAgeAtPOS
#     - MasVnrArea
#     - TotalBsmtSF
#     - 1stFlrSF
#     - GrLivArea 
#     - FullBath
#     - Fireplaces
#     - GarageYrBlt
#     - GarageCars
#     - GarageArea
#     
# _All these are out of the Numerical Features._

# +
# Re-Inferring the results from the heatmap using the Correlation table

hc = house.corr()
SP = hc['SalePrice']

# checking for Important variables (iv) - with pearson value > abs(0.3)
iv = hc.loc[ (abs(hc['SalePrice']) > abs(0.3)) & (hc.index != 'SalePrice'),'SalePrice'].sort_values(ascending=False)
iv

# +
import math

l = len(iv.index)
b = 3
a = math.ceil(l/b)
c = 1
plt.figure(figsize=(18,22))

for i in iv.index:
    plt.subplot(a,b,c);
    sns.regplot(data=house, x= i, y= 'SalePrice');
    c += 1

plt.show();
# -

# - **Inference**
#     - _Most of the variables are continuous except few like GarageCars, TotRmsAbvGrd, FullBath, FirePlaces._
#     - _The **continuous variables (Independent Variables)** in the above plots are **fairly Linearly related** with the **Target Variable, SalePrice**._
#     - Hence, we can safely **perform LINEAR REGRESSION.**

# Less Important variables (liv) - Derived from the corr() table
liv = hc.loc[ (abs(hc['SalePrice']) <= abs(0.3)) & (hc.index != 'SalePrice'),'SalePrice'].sort_values(ascending=False)
liv

# +
l = len(liv.index)
b = 3
a = math.ceil(l/b)
c = 1
plt.figure(figsize=(18,22))

for i in liv.index:
    plt.subplot(a,b,c)
    sns.regplot(data=house, x= i, y= 'SalePrice')
    c += 1

plt.show();
# -

# - **Inference**
#     - _Most of the variables are **actually Categorical**._
#     - _The **continuous variables (Independent Variables)** in the above plots have **poor Linear relation** with the **Target Variable, SalePrice**._
#     - _Hence, we can safely **drop the continuous variables from the above plot.**_
#     - _I would further analyze the actually categorical variables like **Number of Bathrooms, Bedrooms or Kitchen.**_

# +
# Dropping the poorly related continuous independent variables

house.drop(columns=['LotArea','BsmtUnfSF','ScreenPorch','PoolArea','3SsnPorch','BsmtFinSF2','MiscVal','LowQualFinSF','EnclosedPorch'], inplace=True)
# -

lst = ['HalfBath','BsmtFullBath','BedroomAbvGr','BsmtHalfBath','IsReMod','KitchenAbvGr']
for i in lst:
    house[i].value_counts()

# - **Inference**
#     - 'BsmtHalfBath','IsReMod','KitchenAbvGr' seems to be skewed.
#     - To anlyze more, converting to category

house[lst] = house[lst].astype(object)
house[lst].info()

iv_lst = ['FullBath','TotRmsAbvGrd','Fireplaces','GarageCars']
house[iv_lst] = house[iv_lst].astype(object)
house[iv_lst].info()

# ### Analysing the categorical variables

# Selecting only the variables having categorical values
house_cat = house.select_dtypes(exclude='number')
house_cat.head()


# +

## Show labels in bar plots - copied from https://stackoverflow.com/a/48372659
def showLabels(ax, d=None):
    plt.margins(0.2, 0.2)
    rects = ax.patches
    i = 0
    locs, labels = plt.xticks() 
    counts = {}
    if not d is None:
        for key, value in d.items():
            counts[str(key)] = value

    # For each bar: Place a label
    for rect in rects:
        # Get X and Y placement of label from rect.
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Number of points between bar and label. Change to your liking.
        space = 5
        # Vertical alignment for positive values
        va = 'bottom'

        # If value of bar is negative: Place label below bar
        if y_value < 0:
            # Invert space to place label below
            space *= -1
            # Vertically align label at top
            va = 'top'

        # Use Y value as label and format number with one decimal place
        if d is None:
            label = "{:.1f}".format(y_value)
        else:
            try:
                label = "{:.1f}".format(y_value) + "\nof " + str(counts[str(labels[i].get_text())])
            except:
                label = "{:.1f}".format(y_value)
        
        i = i+1

        # Create annotation
        plt.annotate(
            label,                      # Use `label` as label
            (x_value, y_value),         # Place label at end of the bar
            xytext=(0, space),          # Vertically shift label by `space`
            textcoords="offset points", # Interpret `xytext` as offset in points
            ha='center',                # Horizontally center label
            va=va)                      # Vertically align label differently for
                                        # positive and negative values.


# -

# This user-defined function plots the distribution of target column, and its boxplot against loan_status column
def plot_distribution(var):
    plt.figure(figsize=(18,11))
    plt.subplot(1, 3, 1)
    ser = (house[var].value_counts(normalize=True)*100)
    ax = ser.plot.bar(color=sns.color_palette("pastel", 10))
    showLabels(ax);
    plt.subplot(1, 3, 2)
    ser = house[var].value_counts()
    ax = ser.plot.bar(color=sns.color_palette("pastel", 10))
    showLabels(ax);
    #ax = sns.histplot(data=house, x=var, kde=False)
    plt.subplot(1, 3, 3)
    ax = sns.boxplot(x=var, y= 'SalePrice', data=house, order = ser.index,  palette="pastel")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
#     ax.set_xticks(rotation=60)
    plt.show()


for i in house_cat.columns:
    plot_distribution(i)

# +
# From analysing the above plots, there are few skewed categorical variables. - So, Dropping them
skwd_cat_vars = ['Street','Utilities','Condition2','RoofMatl','BsmtCond','Heating','Functional']

house.drop(columns=skwd_cat_vars,inplace=True)

# +
# for i in house_cat[lst]:
#     plot_distribution(i)
# -

# #### Combining minor categories in within categorical variables

# +
# GarageQual GarageCond IsReMod
# -

lst = ['Electrical','BsmtHalfBath','PavedDrive']
for i in lst:
    house[i].value_counts()

house['Electrical_SBrkr'] = np.where(house['Electrical'] == 'SBrkr', 1,0)

house['IsBsmtHalfBath'] = np.where(house['BsmtHalfBath'].astype(str).str.strip() == '0', 0,1)

house['IsFullyPavedDrive'] = np.where(house['PavedDrive'].astype(str).str.strip() == 'Y', 1,0)

house.drop(columns=lst, inplace=True)

# +
lst = ['Electrical_SBrkr','IsFullyPavedDrive','IsBsmtHalfBath']

# changing type
house[lst] = house[lst].astype(object)

for i in lst:
    house[i].value_counts()

# -

# ### Analyzing the target variable - SalePrice

house.SalePrice.isnull().sum()

house.SalePrice.describe()

plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
sns.boxplot(data=house,y='SalePrice',palette='pastel');
plt.subplot(1,2,2)
sns.histplot(data=house,x='SalePrice',kde=True,palette='pastel');

# - The SalePrice is Right-Skewed.
# - We need to fix this as the Regression line will deviate bcoz of outliers.
# - Probable ways could be:
#     - Capping the values
#     - Dropping the Outliers
# - The above techniques NOT PREFERRED, as it will cause LR Model to not predict values in higher range.
# - This would be fixed with the help of Transformation. - _**Log Transformation**_ of the target variable.

# +
# Applying the log transformation technique on the SalePrice column to convert into a normal distributed data
house['SalePriceLog'] = np.log(house['SalePrice'])

# Dropping SalePrice
house.drop(columns='SalePrice',inplace=True)
# -

plt.figure(figsize=(15,7))
plt.subplot(1,2,1)
sns.boxplot(data=house,y='SalePriceLog',palette='pastel');
plt.subplot(1,2,2)
sns.histplot(data=house,x='SalePriceLog',kde=True,palette='pastel');

# creating a new dataframe
house_df = house.copy()
house_df.shape

# ## 4. Model Building and Data Preparation

num_varlist = house_df.select_dtypes(include='number').columns
num_varlist

# - I would apply Scaling on these numerical features.

cat_varlist = house_df.select_dtypes(exclude='number').columns
cat_varlist

# - These catgorical features need to be handled in two parts.
#     - 1. Nominal variables : Directly encode them using pd.get_dummies()
#     - 2. Ordinal variables : Mapping them using map() or applymap()

# +
# Ordinal - ['ExterQual', 'ExterCond','BsmtQual','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'KitchenQual', 'GarageQual', 'GarageCond']

# Nominal - ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig','LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
# 'MasVnrType', 'Foundation', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
# 'GarageType', 'GarageFinish', 'GarageCars', 'SaleType', 'SaleCondition', 'CentralAir']
# -

# ### Encoding/Dummy creation

# +
# List of variables to map

ord_varlist =  ['ExterQual', 'ExterCond', 'HeatingQC', 'KitchenQual']

# Defining the map function
def binary_map(x):
    return x.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4})

# Applying the function to the house_df
house_df[ord_varlist] = house_df[ord_varlist].apply(binary_map)

# +
# List of variables to map

ord_varlist =  ['BsmtQual', 'GarageQual', 'GarageCond']

# Defining the map function
def binary_map(x):
    return x.map({'Po': 0, 'Fa': 1, 'TA': 2, 'Gd': 3, 'Ex': 4, 'None': -1})

# Applying the function to the house_df
house_df[ord_varlist] = house_df[ord_varlist].apply(binary_map)

# +
# List of variables to map

ord_varlist =  ['BsmtExposure']

# Defining the map function
def binary_map(x):
    return x.map({'No': 0, 'Mn': 1, 'Av': 2, 'Gd': 3, 'NA': -1})

# Applying the function to the house_df
house_df[ord_varlist] = house_df[ord_varlist].apply(binary_map)

# +

# List of variables to map

ord_varlist =  ['BsmtFinType1', 'BsmtFinType2']

# Defining the map function
def binary_map(x):
    return x.map({'Unf': 0, 'LwQ': 1, 'Rec': 2, 'BLQ': 3, 'ALQ': 4, 'GLQ': 5, 'NA': -1})

# Applying the function to the house_df
house_df[ord_varlist] = house_df[ord_varlist].apply(binary_map)

# +
# List of variables to map

varlist =  ['CentralAir']

# Defining the map function
def binary_map(x):
    return x.map({'N': 1, "Y": 0})

# Applying the function to the housing list
house_df[varlist] = house_df[varlist].apply(binary_map)
# -

house_df[['ExterQual', 'ExterCond','BsmtQual','BsmtExposure','BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual', 'GarageQual', 'GarageCond']].info()

# Nominal Categorical Features list to create dummies
nomin_varlist = ['MSSubClass', 'MSZoning', 'LotShape', 'LandContour', 'LotConfig','LandSlope', 'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd',
'MasVnrType', 'Foundation', 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
'GarageType', 'GarageFinish', 'GarageCars', 'SaleType', 'SaleCondition', 'CentralAir']

# +
# Create the dummy variables for the Nominal categorical features

dummy = pd.get_dummies(house_df[nomin_varlist], drop_first = True)
dummy.shape
dummy.head(4)

# +
# Dropping the original categorical features

house_df.drop(nomin_varlist,axis=1,inplace=True)

# +
# Adding the dummy features to the original house_df dataframe

house_df = pd.concat([house_df,dummy], axis=1)
# -

house_df.shape

# ### Splitting train and test set

# +
from sklearn.model_selection import train_test_split

house_df_train, house_df_test = train_test_split(house_df, train_size=0.7, test_size=0.3, random_state=100)
# -

house_df_train.shape
house_df_test.shape

# ### Scaling the Numerical features
#
# Machine learning algorithm just sees number — if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort. So these more significant number starts playing a more decisive role while training the model.

# +
# Aplying MinMaxScaler Scaler

from sklearn.preprocessing import MinMaxScaler

# Creating scaler object
scaler = MinMaxScaler()

# +
# Train set
house_df_train[num_varlist] = scaler.fit_transform(house_df_train[num_varlist])

# Test set
house_df_test[num_varlist] = scaler.transform(house_df_test[num_varlist])
# -

# ### Splitting X (predictor) and y (target) in train set

y_train = house_df_train.pop('SalePriceLog')
y_train.shape
X_train = house_df_train
X_train.shape

y_test = house_df_test.pop('SalePriceLog')
y_test.shape
X_test = house_df_test
X_test.shape

# ### Model Building

# Importing LinearRegression and RFE
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

# Creating the linear regression object
lr = LinearRegression()

X_train.isnull().sum().any()

# Applying the fit
lr.fit(X_train, y_train)

# Checking the model cefficients
lr.intercept_
lr.coef_

# #### Distribution of the Error terms
# - **_Residual Analysis needs to be done to validate assumptions of the model, and hence the reliability for inference._**
#
# - We need to check if the error terms are also normally distributed (which is one of the major assumptions of linear regression).
# - Plotting a histogram of the error terms and see what it looks like.

# +
y_train_pred = lr.predict(X_train)
# y_train_pred.head()

# Calculating the residuals
residuals = (y_train - y_train_pred)

# Plot the histogram of the error terms/residuals
plt.figure(figsize=(10,6))
sns.histplot(residuals, stat="density", kde=True, color='#d62728')
plt.title('Residuals Analysis', fontsize = 24)                 # Plot heading 
plt.xlabel('Errors / Residuals', fontsize = 12);                    # X-label
# -

# - Residuals are left skewed, clearly hinting at the outiers.

from scipy import stats
stats.probplot(residuals, dist="norm", plot=plt)

# - Errors are NOT NORMALLY DISTRIBUTED.

# Visualizing the residuals and predicted value on train set
# plt.figure(figsize=(25,12))
sns.jointplot(x = y_train_pred, y = residuals, kind='reg', color='#d62728')
plt.title('Residuals of Linear Regression Model', fontsize = 20, pad = 100) # Plot heading 
plt.xlabel('Predicted Value', fontsize = 12)                     # X-label
plt.ylabel('Residuals', fontsize = 12);        

y_test_pred = lr.predict(X_test)

# +
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# R2 scroe on train data
r_squared = r2_score(np.exp(y_train), np.exp(y_train_pred))
r_squared

# R2 scroe on test data
r_squared = r2_score(y_test, y_test_pred)
r_squared

#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_test_pred))
# -





# ### RFE

# +
# Running RFE

# Create the RFE object
rfe = RFE(lr, n_features_to_select = 50)

rfe = rfe.fit(X_train, y_train)

# +
# Features with rfe.support_ values

list(zip(X_train.columns,rfe.support_,rfe.ranking_))

# +
# Creating a list of rfe supported features
feats = X_train.columns[rfe.support_]
feats

# Creating a list of non-supported rfe features
drop_feats = X_train.columns[~rfe.support_]
drop_feats

# +
# Creating a dataframe with only important features, ranked by RFE method - Train set
X_train_rfe = X_train[feats]
X_train_rfe.shape

# Creating a dataframe with only important features, ranked by RFE method - Test set
X_test_rfe = X_test[feats]
X_test_rfe.shape
# -

# ## Ridge Regression

# +
# Importing libraries

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import sklearn.metrics as metrics
import os

# +
# List of alphas to tune

params = {'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1, 
 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 
 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 20, 50, 100, 500, 1000 ]}


ridge = Ridge()

# cross validation
folds = 5
model_cv = GridSearchCV(estimator = ridge, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            
model_cv.fit(X_train_rfe, y_train)
# -

cv_results = pd.DataFrame(model_cv.cv_results_)
cv_results.head()

# +
# plotting mean test and train scoes with alpha

cv_results['param_alpha'] = cv_results['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results['param_alpha'], cv_results['mean_train_score'])
plt.plot(cv_results['param_alpha'], cv_results['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')
plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show();
# -

print("\n The best estimator across ALL searched params:\n",
          model_cv.best_estimator_)
print("\n The best score across ALL searched params:\n",
          model_cv.best_score_)
print("\n The best parameters across ALL searched params:\n",
          model_cv.best_params_)

# +
# Using the best hyper parameter in the ridge Regression
alpha = .0001
ridge = Ridge(alpha=alpha)

ridge.fit(X_train_rfe, y_train)
ridge.coef_
# -

# predict for the training dataset
y_train_pred = ridge.predict(X_train_rfe)
print('The training accuracy is:')
print(metrics.r2_score(y_true=np.exp(y_train), y_pred=np.exp(y_train_pred)))

# predict for the test dataset
y_test_pred = ridge.predict(X_test_rfe)
print('The testing accuracy is:')
print(metrics.r2_score(y_true=np.exp(y_test), y_pred=np.exp(y_test_pred)))

# model coefficients
cols = X_test_rfe.columns
cols = cols.insert(0, "constant")
model_parameters = list(ridge.coef_)
list(zip(cols, model_parameters))

len(X_test_rfe.columns)

# _**The no of predictors is same as passed in the model after RFE.**_

# ## Now, doubling the hyperparameter value for ridge

# +
alpha_double = .0002
ridge_double = Ridge(alpha=alpha_double)

ridge_double.fit(X_train_rfe, y_train)
ridge_double.coef_
# -

# predict on train
y_train_pred_double = ridge_double.predict(X_train_rfe)
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred_double))

rsquare = metrics.r2_score(y_true=y_train, y_pred=y_train_pred_double)
rssbytss = 1-rsquare
rssbytss

# predict on test
y_test_pred = ridge_double.predict(X_test_rfe)
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

# model coefficients
cols = X_test_rfe.columns
cols = cols.insert(0, "constant")
model_parameters = list(ridge_double.coef_)
ridge_double_list = list(zip(cols, model_parameters))

ridge_double_list

# ## Lasso Regression

# +
# list of alphas to fine tune
params = {'alpha': [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01]}


lasso = Lasso()

# cross validation
model_lasso_cv = GridSearchCV(estimator = lasso, 
                        param_grid = params, 
                        scoring= 'neg_mean_absolute_error', 
                        cv = folds, 
                        return_train_score=True,
                        verbose = 1)            

model_lasso_cv.fit(X_train_rfe, y_train)
# -

cv_results_lasso = pd.DataFrame(model_lasso_cv.cv_results_)
cv_results_lasso.head()

# +
# plotting mean test and train scoes with alpha 
cv_results_lasso['param_alpha'] = cv_results_lasso['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_train_score'])
plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show();
# -

print("\n The best estimator across ALL searched params:\n",
          model_lasso_cv.best_estimator_)
print("\n The best score across ALL searched params:\n",
          model_lasso_cv.best_score_)
print("\n The best parameters across ALL searched params:\n",
          model_lasso_cv.best_params_)

# #### Fitting Lasso

# +
alpha_lasso =0.000001

lasso = Lasso(alpha=alpha_lasso)
        
lasso.fit(X_train_rfe, y_train)
# -

lasso.coef_

# +
# model coefficients

model_parameters = list(lasso.coef_)
cols = X_train_rfe.columns
cols = cols.insert(0, "constant")
model_parameters = list(lasso.coef_)
lasso_list = list(zip(cols, model_parameters))
lasso_list

# +
#List of all predictors with non zero co-efficients
c=0
for i,j in enumerate(lasso_list):
    if(lasso_list[i][1]!=0):
        print(lasso_list[i])
        c+=1

print('\n')        
print('Total predictors used in Lasso ', c) 

# +
lm = Lasso(alpha=0.000001)
lm.fit(X_train_rfe, y_train)

# predict
y_train_pred = lm.predict(X_train_rfe)
print('The training accuracy is:')
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test_rfe)
print('The test accuracy is:')
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# -

# ### Now, doubling the alpha for Lasso

# +
lm = Lasso(alpha=0.000002)
lm.fit(X_train_rfe, y_train)

# predict
y_train_pred = lm.predict(X_train_rfe)
print('The training accuracy is:')
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test_rfe)
print('The test accuracy is:')
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))

# +
# model coefficients

model_parameters = list(lasso.coef_)
cols = X_train_rfe.columns
cols = cols.insert(0, "constant")
model_parameters = list(lasso.coef_)
lasso_list = list(zip(cols, model_parameters))
lasso_list

# +
#List of all predictors with non zero co-efficients
c=0
for i,j in enumerate(lasso_list):
    if(lasso_list[i][1]!=0):
        print(lasso_list[i])
        c+=1

print('\n')        
print('Total predictors used in Lasso ', c) 
# -

# ### Removal of Top 5 predictors from Lasso

top5 = pd.DataFrame(lasso_list)
top5.columns = ['Variable', 'Coeff']
# Sorting the coefficients in ascending order
top5 = top5.drop(index=0,axis=0).sort_values((['Coeff']), axis = 0, ascending = False)
# top5
top5.head(5)

top5_list = list(top5.head(5)['Variable'])
#type(top5_list)
top5_list

X_train_rfe2 = X_train_rfe.drop(columns=top5_list)

X_test_rfe2 = X_test_rfe.drop(columns=top5_list)

model_lasso_cv.fit(X_train_rfe2, y_train)

cv_results_lasso = pd.DataFrame(model_lasso_cv.cv_results_)

# +
# plotting mean test and train scoes with alpha 
cv_results_lasso['param_alpha'] = cv_results_lasso['param_alpha'].astype('float32')

# plotting
plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_train_score'])
plt.plot(cv_results_lasso['param_alpha'], cv_results_lasso['mean_test_score'])
plt.xlabel('alpha')
plt.ylabel('Negative Mean Absolute Error')

plt.title("Negative Mean Absolute Error and alpha")
plt.legend(['train score', 'test score'], loc='upper right')
plt.show();
# -

print("\n The best estimator across ALL searched params:\n",
          model_lasso_cv.best_estimator_)
print("\n The best score across ALL searched params:\n",
          model_lasso_cv.best_score_)
print("\n The best parameters across ALL searched params:\n",
          model_lasso_cv.best_params_)

# +
alpha_lasso =0.0001

lasso = Lasso(alpha=alpha_lasso)
        
lasso.fit(X_train_rfe2, y_train)

# +
# model coefficients

model_parameters = list(lasso.coef_)
cols = X_train_rfe2.columns
cols = cols.insert(0, "constant")
model_parameters = list(lasso.coef_)
lasso_list = list(zip(cols, model_parameters))
lasso_list

# +
#List of all predictors with non zero co-efficients
c=0
for i,j in enumerate(lasso_list):
    if(lasso_list[i][1]!=0):
        print(lasso_list[i])
        c+=1

print('\n')        
print('Total predictors used in Lasso ', c) 
# -

top5 = pd.DataFrame(lasso_list)
top5.columns = ['Variable', 'Coeff']
# Sorting the coefficients in ascending order
top5 = top5.drop(index=0,axis=0).sort_values((['Coeff']), axis = 0, ascending = False)
# top5
top5.head(5)

top5_list = list(top5.head(5)['Variable'])
#type(top5_list)
top5_list

# +
lm = Lasso(alpha=0.0001)
lm.fit(X_train_rfe2, y_train)

# predict
y_train_pred = lm.predict(X_train_rfe2)
print('The training accuracy is:')
print(metrics.r2_score(y_true=y_train, y_pred=y_train_pred))
y_test_pred = lm.predict(X_test_rfe2)
print('The test accuracy is:')
print(metrics.r2_score(y_true=y_test, y_pred=y_test_pred))
# -

# #### Total predictors used in Lasso  33


