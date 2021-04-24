#!/usr/bin/env python
# coding: utf-8

# # Identity
# Assume user downloaded archive.zip from Kaggle, renamed the file BuildingData.zip, and stored the file in the data subdirectory. Assume the zip file contains the weather.csv file.
# 
# This notebook uses a naive model to establish a baseline forecast accuracy. The naive model says energy at time t equals weather at time t-1, scaled by some global conversion factor:
# 
# $energy_{t} = factor * weather_{t-1}$
# 
# This notebook produced the numbers summarized in Report 1, Table I, row="naive".

# In[1]:


DATAPATH=''
try:
    # On Google Drive, set path to my drive / data directory.
    from google.colab import drive
    IN_COLAB = True
    PATH='/content/drive/'
    drive.mount(PATH)
    DATAPATH=PATH+'My Drive/data/'  # must end in "/"
except:
    # On home computer, set path to local data directory.
    IN_COLAB = False
    DATAPATH='data/'  # must end in "/"

ZIP_FILE='BuildingData.zip'
ZIP_PATH = DATAPATH+ZIP_FILE
STEAM_FILE='steam.csv'
WEATHER_FILE='weather.csv'
MODEL_FILE='Model'  # will be used later to save models


# In[2]:


from os import listdir
import csv
from zipfile import ZipFile
import numpy as np
import pandas as pd
import scipy # mean
from scipy import stats  # mode

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from matplotlib import colors
mycmap = colors.ListedColormap(['red','blue'])  # list color for label 0 then 1
np.set_printoptions(precision=2)


# In[3]:


def read_zip_to_panda(zip_filename,csv_filename):
    zip_handle = ZipFile(zip_filename)
    csv_handle = zip_handle.open(csv_filename)
    panda = pd.read_csv(csv_handle)
    return panda
def fix_date_type(panda):
    # Convert the given timestamp column to the pandas datetime data type.
    panda['timestamp'] = pd.to_datetime(panda['timestamp'], infer_datetime_format = True)
    indexed = panda.set_index(['timestamp'])
    return indexed
def get_site_timeseries(panda,site):
    # Assume the panda dataframe has a datetime column.
    # (If not, call fix_date_type() before this.)
    # Extract the timeseries for one site.
    # Convert the datetime column to a DatetimeIndex.
    site_df = panda[panda['site_id']==site]
    temp_col = site_df['date']
    temp_val = temp_col.values
    temp_ndx = pd.DatetimeIndex(temp_val)
    dropped = site_df.drop('date',axis=1)
    panda = dropped.set_index(temp_ndx)
    return panda


# In[4]:


SITE = 'Eagle'
METER = 'steam'
PREDICTOR_VARIABLE = 'airTemperature' 
PREDICTED_VARIABLE = 'steam'  


# In[5]:


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)
stm_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
stm_df = fix_date_type(stm_df)
site_specific_weather = wet_df.loc[wet_df['site_id'] == SITE]
all_buildings = [x for x in stm_df.columns if x.startswith(SITE)]


# In[6]:


DOWNSAMPLE = False   # if true, use 1 time per day, else 24 times per day
STEPS_HISTORY = 1 
STEPS_FUTURE =  1    
def smooth(df):
    # For smoothing the 24 hour cycle, we do not want exponential smoothing.
    smoothed = None
    if DOWNSAMPLE:
        # This alternate method samples down to 1/24 time steps.
        smoothed = df.resample("24H").mean() 
    else:
        # This method does not reduce the number of time steps.
        # Note the first 23 measurements get set to Nan.
        smoothed=df.rolling(window=24).mean()
        smoothed=smoothed[24:]
    return smoothed

# Correlation is low when buildings have many NaN and 0 meter readings.
# We will ignore buildings that have >max bad meter readings.
def is_usable_column(df,column_name):
    MAX_BAD = 500 
    bad = df[column_name].isin([0]).sum()
    return bad<=MAX_BAD

def prepare_for_learning(df):
    X=[]
    y=[]
    predictor_series = df[PREDICTOR_VARIABLE].values
    predicted_series = df[PREDICTED_VARIABLE].values
    for i in range(STEPS_HISTORY,len(df)-STEPS_FUTURE):
        one_predictor = predictor_series[i-STEPS_HISTORY:i]
        one_predicted = predicted_series[i:i+STEPS_FUTURE]
        X.append(one_predictor)
        y.append(one_predicted)
    return X,y  


# In[7]:


cors = []
# Test on only Peter just during code development
for BLDG in all_buildings:
    # Get steam usage for one building.
    bldg_specific_steam = stm_df[[BLDG]]
    # Concatenate steam usage with weather.
    one_bldg_df = pd.concat([bldg_specific_steam,site_specific_weather],axis=1)
    # Drop the site, which is constant (we selected for one site).
    one_bldg_df = one_bldg_df.drop(['site_id'],axis=1)
    # The original steam table used column name = building name.
    # We are processing one building, so rename to the column 'steam'.
    one_bldg_df = one_bldg_df.rename(columns={BLDG : METER})
    # In order to filter bad buildings, count sum of NaN + zero.
    one_bldg_df = one_bldg_df.fillna(0)
    print(BLDG)
    
    if is_usable_column(one_bldg_df,METER):
        one_bldg_df = smooth(one_bldg_df) # moving average: 24hr
        X,y = prepare_for_learning(one_bldg_df)
        # Ideally, split Year1 = train, Year2 = test.
        # Some data is incomplete, so split 1st half and 2nd half.
        split = len(X)//2 
        X_train = X[0:split]
        y_train = y[0:split]
        X_test = X[split:]
        y_test = y[split:]
        factor = np.mean(y_train) / np.mean(X_test)
        #print(factor,"=",np.mean(y_train),"/",np.mean(X_test))
        y_pred = [x*factor for x in X_test]
        # Keep a table for reporting later.
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        mean = one_bldg_df[METER].mean()
        cor = one_bldg_df.corr().loc[PREDICTED_VARIABLE][PREDICTOR_VARIABLE] 
        cors.append([cor,mean,rmse,rmse/mean,BLDG])
        print("Samples:",len(X_train),"Factor:",factor)
        print("RMSE/mean=",rmse/mean)

print()
print("History",STEPS_HISTORY,"Future",STEPS_FUTURE)
print("Column 1: Correlation of",PREDICTED_VARIABLE,"and",PREDICTOR_VARIABLE)
print("          Using one weather feature as leading correlate.")
print("Column 2: Mean usage.")
print("          Using mean to help understand the RMSE.")
print("Column 3: RMSE of LinearRegression(X=Weather, y=Usage).")
print("Column 4: RMSE/mean normalized to help understand RMSE.")
print("Column 5: Building.")
for cor in sorted(cors):
    print("%7.4f %10.2f %10.2f %5.2f   %s"%(cor[0],cor[1],cor[2],cor[3],cor[4]))    


# ### Report 1
# Report 1, Table I, includes the following summary. This is the mean over 16 builings of the normalized RMSE per building.
# 
# Naive model using predictions based on 1 time 1 feature  
# * 1.08 mean RMSE   
# * 0.20 stddev  
# 
# Here are the results omitting outlier building Wesley.
# * 1.12 mean RMSE   
# * 0.13 stddev  
# 

# In[ ]:




