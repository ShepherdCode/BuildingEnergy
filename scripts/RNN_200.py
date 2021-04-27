#!/usr/bin/env python
# coding: utf-8

# # RNN 
# Discovered and fixed bug in prepare() where the y_train for each sample covered the same (past) days as the X_train. The code worked for predicting one time point into the future, as presented in Report 1, but not multiple time points into the future.
# 
# Discovered that the model merely predicts the building's mean steam for every future time point. More epochs of training merely brings the mean closer to true. Examples of (epochs,pred_mean) where true mean is 82: (2,2) (10,20) (20,50) (50,83) (100,86).

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

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.losses import MeanSquaredError

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


EPOCHS=50
SITE = 'Eagle'
METER = 'steam'
PREDICTORS = ['cloudCoverage', 'airTemperature', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed']
#PREDICTORS.append('steam')
print("PREDICTORS=",len(PREDICTORS),PREDICTORS)
NUM_PREDICTORS = len(PREDICTORS)  
PREDICTED_VARIABLE = 'steam'  
STEPS_HISTORY = 24 
STEPS_FUTURE =  24    


# In[5]:


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)
stm_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
stm_df = fix_date_type(stm_df)
site_specific_weather = wet_df.loc[wet_df['site_id'] == SITE]
all_buildings = [x for x in stm_df.columns if x.startswith(SITE)]


# In[6]:


# Correlation is low when buildings have many NaN and 0 meter readings.
# We will ignore buildings that have >max bad meter readings.
def is_usable_column(df,column_name):
    MAX_BAD = 500 
    bad = df[column_name].isin([0]).sum()
    return bad<=MAX_BAD

def prepare_for_learning(df):
    num_samples = len(df) - STEPS_FUTURE - STEPS_HISTORY
    X_shape = (num_samples,STEPS_HISTORY,NUM_PREDICTORS)
    X=np.zeros(X_shape)
    Y_shape = (num_samples,STEPS_FUTURE)
    y=np.zeros(Y_shape)
    predictor_series = df[PREDICTORS].values  # e.g. all weather values
    predicted_series = df[PREDICTED_VARIABLE].values  # e.g. all meter readings
    
    for sam in range (0,num_samples): # Loop over all 1000 samples
        # This is one array of weather for previous 24 time periods
        one_sample = predictor_series[sam:sam+STEPS_HISTORY]
        # Loop over all 24 time periods
        for time in range (0,STEPS_HISTORY): # In 1 sample, loop over 24 time periods
            one_period = one_sample[time]
            for feat in range (0,NUM_PREDICTORS): # In 1 time period, loop over 8 weather metrics
                X[sam,time,feat] = one_period[feat]
        for time in range (0,STEPS_FUTURE):  
            y[sam,time]=predicted_series[sam+STEPS_HISTORY+time]
    return X,y 


# In[7]:


def make_RNN():
    rnn = Sequential([
        SimpleRNN(8,return_sequences=True, 
                  input_shape=(STEPS_HISTORY,NUM_PREDICTORS)), 
        SimpleRNN(8,return_sequences=False),
        Dense(STEPS_FUTURE)
    ])
    rnn.compile(optimizer='adam',loss=MeanSquaredError())
    return rnn


# In[8]:


cors = []
for BLDG in ['Eagle_lodging_Edgardo']:  ### all_buildings:
    print("Building",BLDG)
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
        X,y = prepare_for_learning(one_bldg_df)
        split = len(X)//2   # year 1 vs year 2
        X_train = np.asarray(X[0:split])
        y_train = np.asarray(y[0:split])
        X_test = np.asarray(X[split:])
        y_test = np.asarray(y[split:])

        model = make_RNN()
        print(model.summary())
        print("Example X train:\n",X_train[210].astype(int))
        print("Example y train:\n",y_train[210].astype(int))
        model.fit(X_train,y_train,epochs=EPOCHS)
        y_pred = model.predict(X_test)
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        # Keep a table for reporting later.
        mean = one_bldg_df[METER].mean()
        cors.append([mean,rmse,rmse/mean,BLDG])
        print("mean,rmse,rmse/mean,bldg:",mean,rmse,rmse/mean,BLDG)
        print("Example prediction:\n",y_pred[210].astype(int))
        print("Example truth:\n",y_test[210].astype(int))

print("History",STEPS_HISTORY,"Future",STEPS_FUTURE)
print("Column 1: Mean usage.")
print("Column 2: RMSE of LinearRegression(X=Weather, y=Usage).")
print("Column 3: RMSE/mean normalized to help understand RMSE.")
print("Column 4: Building.")
for cor in sorted(cors):
    print("%10.2f %10.2f %5.2f   %s"%(cor[0],cor[1],cor[2],cor[3]))    


# In[ ]:





# In[ ]:




