#!/usr/bin/env python
# coding: utf-8

# # LSTM 
# Here, use 8 weather features.

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


SITE = 'Eagle'
METER = 'steam'
BLDG = 'Eagle_education_Peter'   # one example
PREDICTOR_VARIABLE = 'dewTemperature'  # for starters
PREDICTORS = ['cloudCoverage', 'airTemperature', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed']
print("PREDICTORS=",len(PREDICTORS),PREDICTORS)
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
STEPS_HISTORY = 24 
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
    num_samples = len(df) - STEPS_FUTURE - STEPS_HISTORY
    num_predictors = len(PREDICTORS)
    X_shape = (num_samples,STEPS_HISTORY,num_predictors)
    X=np.zeros(X_shape)
    Y_shape = (num_samples,STEPS_FUTURE)
    y=np.zeros(Y_shape)
    predictor_series = df[PREDICTORS].values  # e.g. all weather values
    predicted_series = df[PREDICTED_VARIABLE].values  # e.g. all meter readings
    
    for x0 in range (0,num_samples): # Loop over all 1000 samples
        # This is one array of weather for previous 24 time periods
        one_sample = predictor_series[x0:x0+STEPS_HISTORY]
        one_label =  predicted_series[x0+STEPS_HISTORY:x0+STEPS_FUTURE]
        # Loop over all 24 time periods
        for x1 in range (0,STEPS_HISTORY): # In 1 sample, loop over 24 time periods
            one_period = one_sample[x1]
            for x2 in range (0,num_predictors): # In 1 time period, loop over 8 weather metrics
                one_predictor = one_period[x2]
                X[x0,x1,x2] = one_predictor
        y[x0]=predicted_series[x0:x0+STEPS_FUTURE]
    return X,y 


# In[7]:


num_predictors = len(PREDICTORS)  # e.g. 8 weather features
def make_RNN():
    rnn = Sequential([
        SimpleRNN(20,return_sequences=True, 
                  input_shape=(STEPS_HISTORY,num_predictors)), 
        SimpleRNN(10,return_sequences=False),
        Dense(STEPS_FUTURE)   
    ])
    rnn.compile(optimizer='adam',loss=MeanSquaredError())
    return rnn


# In[8]:


cors = []
EPOCHS=25
# Test on only Peter just during code development
for BLDG in all_buildings:
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
        one_bldg_df = smooth(one_bldg_df) 
        X,y = prepare_for_learning(one_bldg_df)
        # Ideally, split Year1 = train, Year2 = test.
        # Some data is incomplete, so split 1st half and 2nd half.
        split = len(X)//2 
        X_train = np.asarray(X[0:split])
        y_train = np.asarray(y[0:split])
        X_test = np.asarray(X[split:])
        y_test = np.asarray(y[split:])
        print("Train on",len(X_train),"samples such as",X_train[100][0])
        print("Predict",len(y_train),"labels such as",y_train[100])

        model = make_RNN()
        print(model.summary())
        model.fit(X_train,y_train,epochs=EPOCHS)
        y_pred = model.predict(X_test)
        # Compare. Solve the problem that predict.shape != truth.shape 
        ##print(" before ytestshape",y_test.shape,"ypredshape",y_pred.shape)
        #nsamples, nsteps, ndim = y_test.shape
        #y_test = y_test.reshape(nsamples,nsteps*ndim)
        #nsamples, nsteps, ndim = y_pred.shape
        #y_pred = y_pred.reshape(nsamples,nsteps*ndim)
        ##print(" after ytestshape",y_test.shape,"ypredshape",y_pred.shape)
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        # Keep a table for reporting later.
        mean = one_bldg_df[METER].mean()
        cor = one_bldg_df.corr().loc[PREDICTED_VARIABLE][PREDICTOR_VARIABLE] 
        cors.append([cor,mean,rmse,rmse/mean,BLDG])
        print("cor,mean,rmse,rmse/mean,bldg:",cor,mean,rmse,rmse/mean,BLDG)

        ## break   ## REMOVE THIS LINE TO LOOP OVER BUILDINGS!
        
if True:
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


# In[8]:




