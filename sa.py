# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 13:47:45 2022

@author: seyma
"""

import numpy as np
import pandas as pd
import altair as alt
from matplotlib import pyplot as plt
import seaborn as sns
import math	
from math import sqrt
import os
import scipy
from scipy.stats import kurtosis
from sklearn.metrics import mean_squared_error
import ruptures as rpt
from prophet import Prophet
from fbprophet.plot import plot_plotly, add_changepoints_to_plot
from scipy.stats import skew
import xgboost as xgb
import catboost as cb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import statistics
import boto3

df = pd.read_csv("veriler_son.csv")
df.drop(["MID","t(Meas.)","t(Trans.)","t(Calc.)","T(Complete)","IdentityNo 1","Q-Action","Date"],axis=1,inplace=True)
df.replace({"#.##":np.nan,"-.--":np.nan},inplace=True)
#df=df.set_index(['ID'])
df.fillna(0,inplace=True)
sayi = []
for i in range(len(df)):
    sayi.append(i)        
df['ID_']=sayi


def make_float(col_name):
    if df[col_name].dtypes=="O":
        df[col_name] = df[col_name].astype(float)

for col_name in df.columns:
    # print(df.loc[df[col_name]])
    make_float(col_name)
    
df_regresyon=pd.DataFrame()

df_regresyon['ID_']=df['ID_'][7000:7050]
df_regresyon['10920Z_Z (3)']=df['10920Z_Z (3)'][7000:7050]

df_regresyon['14002Z_Z (3)']=df['14002Z_Z (3)'][7000:7050]
df_regresyon['14100XYZ_Z (3)']=df['14100XYZ_Z (3)'][7000:7050]
df_regresyon['14102XYZ_Z (3)']=df['14102XYZ_Z (3)'][7000:7050]


#df_regresyon['14102XYZ_Y (3)']=df['14102XYZ_Y (3)'][2400:2470]

plt.plot(df_regresyon['ID_'],df_regresyon['10920Z_Z (3)'],color='red',label='10920Z_Z (3)')
plt.legend()
plt.plot(df_regresyon['ID_'],df_regresyon['14002Z_Z (3)'],color='blue',label='14002Z_Z (3)')
plt.legend()
plt.plot(df_regresyon['ID_'],df_regresyon['14100XYZ_Z (3)'],color='orange',label='14100XYZ_Z (3)')
plt.legend()
plt.plot(df_regresyon['ID_'],df_regresyon['14102XYZ_Z (3)'],color='green',label='14102XYZ_Z (3)')
plt.legend()
plt.show()
# =============================================================================
# plt.plot(df_regresyon['ID_'],df_regresyon['14100XYZ_Z (3)'],color='blue',label='14100XYZ_Z (3)')
# plt.legend()
# plt.show()
# plt.plot(df_regresyon['ID_'],df_regresyon['14102XYZ_Z (3)'],color='green',label='14102XYZ_Z (3)')
# plt.legend()
# plt.show()
# =============================================================================
# # ============================================================================
# x=np.array([1,2,3,4,5,6])
# y=np.array([2,4,6,8,10,12])
# =============================================================================
def cross_corr_norm(array1, array2):
    try :
        corr =((5*sum(array1 * array2))-(sum(array1)*sum(array2)))/sqrt( ((5*sum(array1 * array1))-((sum(array1)) *( sum(array1))))*((5*sum(array2 * array2))-((sum(array2)) *( sum(array2)))))
    except ZeroDivisionError as err:
        print('Handling run-time error:', err) 
        corr=0
    return corr


#cross_corr_norm(df['10920Z_Z (3)'],df['14102XYZ_Y (3)'])
row = list()
row1=list()
row2=list()
#for x in range(30,len(df.index)):
for x in range(5,8000):
    df_w30 = df.iloc[x:x-5:-1]
    cross_correlation=cross_corr_norm(df_w30['10920Z_Z (3)'],df_w30['14100XYZ_Z (3)'])
    row.append(cross_correlation)
    cross_correlation1=cross_corr_norm(df_w30['10920Z_Z (3)'],df_w30['14102XYZ_Z (3)'])
    row1.append(cross_correlation1)
    cross_correlation2=cross_corr_norm(df_w30['10920Z_Z (3)'],df_w30['14002Z_Z (3)'])
    row2.append(cross_correlation2)
       

df_cross=pd.DataFrame({'cross_corr100':row})
df_cross['cross_corr102']=row1
df_cross['cross_corr14']=row2
sayi = []
for i in range(5,8000):
    sayi.append(i)        
df_cross['ID_']=sayi

plt.plot(df_cross['ID_'][7040:7110],df_cross['cross_corr100'][7040:7110],color='red',label='14100XYZ_Z (3)')
plt.legend()
plt.plot(df_cross['ID_'][7040:7110],df_cross['cross_corr102'][7040:7110],color='blue',label='14102XYZ_Z (3)')
plt.legend()
plt.plot(df_cross['ID_'][7040:7110],df_cross['cross_corr14'][7040:7110],color='orange',label='14002Z_Z (3)')
plt.legend()

plt.show()

print(min(df_cross['cross_corr100']))
#plt.plot(df_cross['cross_corr'], color = 'blue')
