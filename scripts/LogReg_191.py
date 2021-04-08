#!/usr/bin/env python
# coding: utf-8

# # Linear Regression, prep for LSTM
# In preparation for LSTM, get lineary regression working.
# Experiment with smoothing and downsampling away the daily cycle.
# Start with univariate e.g. predict future air temp from past air temp.
# 
# Prepare X and y as series of vectors.
# Each past vector in X holds the daily average air temp for previous 7 days.
# Each future in y holds the daily average air temp for next 1 days.
# The past and future durations do not have to be the same.
# Linear regression will view each vector as one point in high-dimensional space.
# Train on pairs of past / future from year 1.
# Test on pairs of past / future from year 2.
# 
# LSTM will view each vector as a time series.

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
BLDG = 'Eagle_education_Peter'   # one example
PREDICTOR_VARIABLE = 'airTemperature'  # for starters
PREDICTED_VARIABLE = 'steam'  # for starters


# In[5]:


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)
stm_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
stm_df = fix_date_type(stm_df)
site_specific_weather = wet_df.loc[wet_df['site_id'] == SITE]
all_buildings = [x for x in stm_df.columns if x.startswith(SITE)]


# In[6]:


DOWNSAMPLE = False   # if true, use 1 time per day, else 24 times per day
STEPS_HISTORY = 24*7 
STEPS_FUTURE =  24    
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
    # This is very slow. Is there a faster way? See...
    # https://stackoverflow.com/questions/27852343/split-python-sequence-time-series-array-into-subsequences-with-overlap
    # X = df.drop(METER,axis=1) # this would use all predictors, just drop the predicted
    X=[]
    y=[]
    predictor_series = df[PREDICTOR_VARIABLE].values
    predicted_series = df[PREDICTED_VARIABLE].values
    for i in range(STEPS_HISTORY,len(df)-STEPS_FUTURE):
        one_predictor = predictor_series[i-STEPS_HISTORY:i]
        one_predicted = predicted_series[i:i+STEPS_FUTURE]
        X.append(one_predictor)
        y.append(one_predicted)
    return X,y  # both are list of dataframe


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
    
    if is_usable_column(one_bldg_df,METER):
        one_bldg_df = smooth(one_bldg_df) # moving average: 24hr
        X,y = prepare_for_learning(one_bldg_df)
        if True:
            #X = one_bldg_df.drop(METER,axis=1)
            #y = one_bldg_df[METER]
            # Ideally, split Year1 = train, Year2 = test.
            # Some data is incomplete, so split 1st half and 2nd half.
            split = len(X)//2 
            X_train = X[0:split]
            y_train = y[0:split]
            X_test = X[split:]
            y_test = y[split:]
            linreg = LinearRegression()
            linreg.fit(X_train,y_train)
            y_pred = linreg.predict(X_test)
            # Keep a table for reporting later.
            rmse = mean_squared_error(y_test,y_pred,squared=False)
            mean = one_bldg_df[METER].mean()
            cor = one_bldg_df.corr().iloc[0][3] # corr(steam,dew_temp)
            cors.append([cor,mean,rmse,rmse/mean,BLDG])
if True:
    print("History",STEPS_HISTORY,"Future",STEPS_FUTURE)
    print("Column 1: Correlation of steam usage to dew temp.")
    print("          Using dew temp as leading weather correlate.")
    print("Column 2: Mean steam usage.")
    print("          Using mean to help understand the RMSE.")
    print("Column 3: RMSE of LinearRegression(X=Weather, y=SteamUsage).")
    print("Column 4: RMSE/mean normalized to help understand RMSE.")
    print("Column 5: Building.")
    for cor in sorted(cors):
        print("%7.4f %10.2f %10.2f %5.2f   %s"%(cor[0],cor[1],cor[2],cor[3],cor[4]))    


# ## Results 1
# Downsample to daily mean. It runs a lot faster this way!
# Use air temp to predict steam usage.
# Use past 7 days to predict next 1 day.
# <pre>
# -0.8895    2032.67     445.92  0.22   Eagle_education_Sherrill
# -0.8563    1635.33     650.65  0.40   Eagle_education_Brooke
# -0.8526    3149.69     841.92  0.27   Eagle_education_Peter
# -0.8412     477.70     156.88  0.33   Eagle_health_Athena
# -0.8203    1197.84     280.65  0.23   Eagle_education_Roman
# -0.8004     121.95      21.84  0.18   Eagle_health_Vincenza
# -0.7994      57.05      19.09  0.33   Eagle_education_Petra
# -0.7740     712.07     236.18  0.33   Eagle_education_Norah
# -0.7628     182.08      71.50  0.39   Eagle_public_Alvin
# -0.7222      81.97      29.20  0.36   Eagle_lodging_Edgardo
# -0.7132      92.83      40.80  0.44   Eagle_lodging_Dawn
# -0.6798     148.51      44.58  0.30   Eagle_education_Teresa
# -0.6778      91.28      33.89  0.37   Eagle_lodging_Trina
# -0.5591     336.36     121.94  0.36   Eagle_office_Francis
# -0.3639     226.25      83.15  0.37   Eagle_education_Will
#  0.7265       0.11       0.02  0.21   Eagle_education_Wesley
#  </pre>

# ## Results 2
# Downsample to daily mean. It runs a lot faster this way! 
# Use air temp to predict steam usage. 
# Use past 7 days to predict future 7 days.
# <pre>
# -0.8895    2032.67     612.02  0.30   Eagle_education_Sherrill
# -0.8563    1635.33     718.11  0.44   Eagle_education_Brooke
# -0.8526    3149.69     915.55  0.29   Eagle_education_Peter
# -0.8412     477.70     173.98  0.36   Eagle_health_Athena
# -0.8203    1197.84     323.40  0.27   Eagle_education_Roman
# -0.8004     121.95      25.13  0.21   Eagle_health_Vincenza
# -0.7994      57.05      21.92  0.38   Eagle_education_Petra
# -0.7740     712.07     278.27  0.39   Eagle_education_Norah
# -0.7628     182.08      73.63  0.40   Eagle_public_Alvin
# -0.7222      81.97      32.82  0.40   Eagle_lodging_Edgardo
# -0.7132      92.83      43.79  0.47   Eagle_lodging_Dawn
# -0.6798     148.51      45.46  0.31   Eagle_education_Teresa
# -0.6778      91.28      37.28  0.41   Eagle_lodging_Trina
# -0.5591     336.36     122.46  0.36   Eagle_office_Francis
# -0.3639     226.25      84.54  0.37   Eagle_education_Will
#  0.7265       0.11       0.02  0.21   Eagle_education_Wesley
#  </pre>

# ## Results 3
# Rolling average replaces each hour with daily mean.  
# Use air temp to predict steam usage. 
# Use past week (24*7 hours) to predict next 1 day (24 hours).
# <pre>
# -0.8877    2030.36     387.60  0.19   Eagle_education_Sherrill
# -0.8545    1634.28     633.41  0.39   Eagle_education_Brooke
# -0.8502    3147.43     806.81  0.26   Eagle_education_Peter
# -0.8403     477.41     150.04  0.31   Eagle_health_Athena
# -0.8186    1197.02     254.84  0.21   Eagle_education_Roman
# -0.7988      56.96      18.32  0.32   Eagle_education_Petra
# -0.7961     121.91      21.13  0.17   Eagle_health_Vincenza
# -0.7722     711.33     218.80  0.31   Eagle_education_Norah
# -0.7596     181.94      70.50  0.39   Eagle_public_Alvin
# -0.7222      81.87      28.14  0.34   Eagle_lodging_Edgardo
# -0.7130      92.73      40.38  0.44   Eagle_lodging_Dawn
# -0.6796     148.51      44.79  0.30   Eagle_education_Teresa
# -0.6741      91.20      32.77  0.36   Eagle_lodging_Trina
# -0.5553     335.96     120.01  0.36   Eagle_office_Francis
# -0.3579     226.07      82.74  0.37   Eagle_education_Will
#  0.7231       0.11       0.02  0.21   Eagle_education_Wesley
#  </pre>
