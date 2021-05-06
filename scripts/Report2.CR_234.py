#!/usr/bin/env python
# coding: utf-8

# # CNN + RNN 
# Compare to RNN 229.
# 

# In[36]:


from os import listdir
import csv
from zipfile import ZipFile
import numpy as np
import pandas as pd
from scipy import stats  # mode

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import SimpleRNN
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.losses import MeanSquaredError
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.layers import TimeDistributed

import matplotlib.pyplot as plt
from matplotlib import colors
mycmap = colors.ListedColormap(['red','blue'])  # list color for label 0 then 1
np.set_printoptions(precision=2)


# In[37]:


# Constants
EPOCHS=50  # use 5 for software testing, 50 for model testing
SITE = 'Eagle'
PREDICTORS = ['hour','month','doy','meter','cloudCoverage', 'airTemperature', 'dewTemperature', 'precipDepth1HR', 'precipDepth6HR', 'seaLvlPressure', 'windDirection', 'windSpeed']
# For our CNN, min predictive features is six.
NUM_PREDICTORS=len(PREDICTORS)
print("PREDICTORS=",NUM_PREDICTORS,PREDICTORS)
PREDICTED_VARIABLE = 'meter'  
STEPS_HISTORY = 24
STEPS_FORWARD = 12 
STEPS_FUTURE =  12 
METER_FILE='steam.csv'
WEATHER_FILE='weather.csv'
EXAMPLE='Eagle_lodging_Edgardo'
SITE_BUILDINGS = None
SMOOTHING_WINDOW=3
SCALING=1
CELLS = 16
FILTERS = 16
WIDTH = 3
STRIDE = (1,1)
INPUT_SHAPE = (STEPS_FORWARD,NUM_PREDICTORS,1) 


# In[38]:


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
MODEL_FILE='Model'  # will be used later to save models


# In[39]:


def scale(df):
    scaler=StandardScaler()
    #scaler=MinMaxScaler()
    scaled=scaler.fit_transform(df.values)
    scaled = pd.DataFrame(scaled,index=df.index,columns=df.columns)
    return scaled


# In[40]:


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


# In[41]:


DATE_PARSE=True  # must be true if we use one of these as predictor
def load_weather_for_site(site):
    wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
    wet_df = fix_date_type(wet_df)
    site_df = wet_df.loc[wet_df['site_id'] == site]
    # Drop the site, which is constant (we selected for one site).
    site_df = site_df.drop(['site_id'],axis=1)
    if DATE_PARSE:
        site_df.insert(0,'hour',0)
        site_df.insert(1,'month',0)
        site_df.insert(2,'doy',0)
        L=len(site_df)
        for i in range(0,L):
            dt=site_df.index[i]
            hour=dt.hour
            month=dt.month
            doy=dt.dayofyear
            site_df.iat[i,0] = hour
            site_df.iat[i,1] = month
            site_df.iat[i,2] = doy
    #if SCALING==1:
    #    site_df = scale(site_df) # could break if any column is empty
    return site_df

one_site_weather = load_weather_for_site(SITE)
one_site_weather.tail()


# In[42]:


def load_meter_for_building(bldg,smooth=0):
    all_df = read_zip_to_panda(ZIP_PATH,METER_FILE)
    all_df = fix_date_type(all_df)
    global SITE_BUILDINGS
    SITE_BUILDINGS = [x for x in all_df.columns if x.startswith(SITE)]
    site_series = all_df[bldg]
    site_df = site_series.to_frame()
    #site_df = all_df.loc[all_df['site_id'] == site]
    # Change column name from building name to meter.
    site_df = site_df.rename(columns={bldg : PREDICTED_VARIABLE})
    if smooth>0:
        site_df = site_df.rolling(smooth).mean()
    return site_df

one_bldg_meter = load_meter_for_building(EXAMPLE)
print(type(one_bldg_meter))
one_bldg_meter.tail()


# In[43]:


# Make X out of weather + meter features, then select wanted columns.
# Make y out of meter features, then select meter column only.
# Apply scaler to X only.
# Pull out time steps for X from the past: STEPS_HISTORY.
# Pull out proper number of time steps for X: STEPS_FORWARD.
# Pull out proper number of time steps for y: STEPS_FUTURE.
# Make X inefficiently with 3-deep nested for loop because
# a) want to replace NaN with previous value
# b) need to add the RGB dimension for CNN.
def prepare_for_learning(wdf,mdf):
    df = pd.concat([wdf,mdf],axis=1)
    if SCALING==1:
        df = scale(df) # could break if any column is empty
    num_samples = len(df) - STEPS_FUTURE - STEPS_HISTORY
    predictor_series = df[PREDICTORS].values  # selected features
    predicted_series = mdf[PREDICTED_VARIABLE].values  # meter
    #
    X_shape = (num_samples,STEPS_FUTURE,NUM_PREDICTORS,1) # RGB = 1
    Y_shape = (num_samples,STEPS_FUTURE)
    X=np.zeros(X_shape)
    y=np.zeros(Y_shape)
    for sam in range (0,num_samples): 
        prev_val = 0
        one_sample = predictor_series[sam:sam+STEPS_FORWARD]
        for time in range (0,STEPS_FORWARD): 
            one_period = one_sample[time]
            for feat in range (0,NUM_PREDICTORS):
                val = one_period[feat]
                if np.isnan(val):
                    val = prev_val
                else:
                    prev_val = val
                X[sam,time,feat,0] = val  # RGB dim = 0
        for time in range (0,STEPS_FUTURE):  
            y[sam,time]=predicted_series[sam+STEPS_HISTORY+time]
    return X,y 
print(one_bldg_meter.head())
X,y = prepare_for_learning(one_site_weather,one_bldg_meter)
print("X shape:",X.shape)
print("y shape:",y.shape)


# In[44]:


print("X columns:",PREDICTORS)
print("X example:\n",X[100].astype(int))
print("y example:\n",y[100].astype(int))


# In[45]:


def make_DNN():
    print("make_DNN")
    print("input shape:",INPUT_SHAPE)
    dnn = Sequential()
    dnn.add(Conv2D( input_shape=INPUT_SHAPE,
            filters=FILTERS,kernel_size=WIDTH,strides=STRIDE,
            activation=None, padding="valid"))
    dnn.add(Conv2D(
            filters=FILTERS,kernel_size=WIDTH,strides=STRIDE,
            activation=None, padding="valid"))
    dnn.add(MaxPooling2D())
    #dnn.add(TimeDistributed(Flatten()))
    dnn.add(Flatten())
    #dnn.add(GRU(CELLS,return_sequences=True))
    #dnn.add(GRU(CELLS,return_sequences=False))
    dnn.add(Dense(STEPS_FUTURE))   
    dnn.compile(optimizer='adam',loss=MeanSquaredError())
    dnn.build(input_shape=INPUT_SHAPE)
    return dnn    


# In[46]:


cors = []
overall = 0
cnt = 0
one_site_weather = load_weather_for_site(SITE)
for BLDG in SITE_BUILDINGS:
    print("Building",BLDG)
    one_bldg_meter = load_meter_for_building(BLDG,SMOOTHING_WINDOW)
    count_bad = one_bldg_meter[PREDICTED_VARIABLE].isna().sum()
    MAX_BAD = 500
    if count_bad<=MAX_BAD:
        # Must get rid of Nan labels, else loss hits NaN during training.
        print(" Count bad values before pseudofill:",count_bad)
        pseudovalue = one_bldg_meter[PREDICTED_VARIABLE].mean()
        one_bldg_meter = one_bldg_meter.fillna(pseudovalue)
        count_bad = one_bldg_meter[PREDICTED_VARIABLE].isna().sum()
        print(" Count bad values after pseudofill:",count_bad)
        # Smoothing window applies to inputs
        X,y = prepare_for_learning(one_site_weather,one_bldg_meter)
        split = len(X)//2   # year 1 vs year 2
        X_train = np.asarray(X[0:split])
        y_train = np.asarray(y[0:split])
        X_test = np.asarray(X[split:])
        # Smoothing does not apply to truth
        one_bldg_meter = load_meter_for_building(BLDG,0)
        one_bldg_meter = one_bldg_meter.fillna(pseudovalue)
        X_raw,y_raw = prepare_for_learning(one_site_weather,one_bldg_meter)
        y_test = np.asarray(y_raw[split:])
        # Train and predict
        model = make_DNN()
        print(model.summary())
        example=411
        print("Example y train:\n",y_train[example].astype(int))
        model.fit(X_train,y_train,epochs=EPOCHS)
        y_pred = model.predict(X_test)
        # Reporting
        rmse = mean_squared_error(y_test,y_pred,squared=False)
        mean = one_bldg_meter[PREDICTED_VARIABLE].mean()
        cors.append([mean,rmse,rmse/mean,BLDG])
        cnt += 1
        print("i,mean,rmse,rmse/mean,bldg:",cnt,mean,rmse,rmse/mean,BLDG)
        overall += rmse/mean
        for hr in range(0,24,2):
            print("Example prediction:\n",hr,y_pred[example+hr].astype(int))


# In[47]:


print("History",STEPS_HISTORY,"Future",STEPS_FUTURE)
print("Column 1: Mean usage.")
print("Column 2: RMSE of LinearRegression(X=Weather, y=Usage).")
print("Column 3: RMSE/mean normalized to help understand RMSE.")
print("Column 4: Building.")
for cor in sorted(cors):
    print("%10.2f %10.2f %5.2f   %s"%(cor[0],cor[1],cor[2],cor[3]))  
overall = overall/cnt
print ("overall = ",overall)  


# In[47]:





# In[47]:




