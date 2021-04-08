#!/usr/bin/env python
# coding: utf-8

# # Eagle 102
# Try forecasting on all buildings (separately) from one site.

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
from pandas.plotting import autocorrelation_plot
from scipy import stats  # mode

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

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
BLDG = 'Eagle_education_Peter'
wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)
stm_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
stm_df = fix_date_type(stm_df)
site_specific_weather = wet_df.loc[wet_df['site_id'] == SITE]
all_buildings = [x for x in stm_df.columns if x.startswith(SITE)]


# In[18]:


cors = []
MAX_BAD = 500 # correlation is higher in buildings without so many NaN and 0
for BLDG in all_buildings:
    bldg_specific_steam = stm_df[[BLDG]]    
    one_bldg_df = pd.concat([bldg_specific_steam,site_specific_weather],axis=1)
    one_bldg_df = one_bldg_df.drop(['site_id'],axis=1)
    one_bldg_df = one_bldg_df.rename(columns={BLDG : METER})
    one_bldg_df = one_bldg_df.fillna(0)
    bad = one_bldg_df[METER].isin([0]).sum()
    if bad<=500:
        mean = one_bldg_df[METER].mean()
        cor = one_bldg_df.corr().iloc[0][3]
        # Linear Regression
        X = one_bldg_df.drop(METER,axis=1)
        y = one_bldg_df[METER].fillna(0)
        split = 900
        X_train = X.iloc[0:split]
        y_train = y.iloc[0:split]
        linreg = LinearRegression()
        linreg.fit(X_train,y_train)
        X_test = X.iloc[split:]
        y_test = y.iloc[split:]
        y_pred = linreg.predict(X_test)
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        cors.append([cor,mean,rmse,rmse/mean,BLDG])

print("dew temp corr, dew temp mean, lin reg RMSE, RMSE/mean, BLDG")
for cor in sorted(cors):
    print("%7.4f %10.2f %10.2f %5.2f   %s"%(cor[0],cor[1],cor[2],cor[3],cor[4]))    


# In[ ]:




