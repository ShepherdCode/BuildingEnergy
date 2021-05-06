#!/usr/bin/env python
# coding: utf-8

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
    DATAPATH='C:/'  # must end in "/"

ZIP_FILE='BuildingData.zip'
ZIP_PATH = DATAPATH+ZIP_FILE
STEAM_FILE='steam.csv'
MODEL_FILE='Model'  # will be used later to save models


# In[ ]:


from os import listdir
import csv
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
#from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller,acf,pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tools.eval_measures import rmse
from math import sqrt

import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib import colors
mycmap = colors.ListedColormap(['red','blue'])  # list color for label 0 then 1
np.set_printoptions(precision=2)


# In[ ]:


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


# In[ ]:


steam_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
steam_df = fix_date_type(steam_df)
steam_df.info()


# In[ ]:


buildings = [c for c in steam_df.columns if 'Eagle' in c]
print(buildings)


# In[ ]:


# Before analyzing the entire dataset, we look at this subset.
SITE = 'Eagle'
METER = 'steam'

stm_df = read_zip_to_panda(ZIP_PATH,STEAM_FILE)
stm_df = fix_date_type(stm_df)
stm_df = stm_df.fillna(4)
#site_specific_weather = stm_df.loc[stm_df['site_id'] == SITE]
all_buildings = [x for x in stm_df.columns if x.startswith(SITE)]


# ## Check Stationarity
# 1. Plotting and print mean and standard deviation
# 2. ADF method

# In[ ]:



for BLDG in all_buildings:
    print("Building",BLDG)
    # Get steam usage for one building.
    bldg_specific_steam = stm_df[BLDG]
    bldg_specific_steam= pd.DataFrame(bldg_specific_steam)
    bldg_specific_steam = bldg_specific_steam.fillna(0)
    #Perform Building Dickey-Fuller test:
    print ('Results of Dickey-Fuller Test: \n the test statistic is less than critical value, reject the null hypothesis')
    dftest = adfuller(bldg_specific_steam, autolag = 'AIC') #AIC gives the information about time series 
    dfoutput= pd.Series (dftest[0:4], index= ['Test Statistic','p-value: \n p-value is smaller than 0.05','#lags used', 'Number of Observations Used'])
    for key, value in dftest [4].items ():
        dfoutput ['Critical Value (%s)' %key] = value

    print (dfoutput)
    #Determine the rolling statistic
    rolmean = bldg_specific_steam.rolling(window = 24).mean()
    rolstd = bldg_specific_steam.rolling(window = 24).std()

    #Checking the Stationarity
    #Plot rolling statistics
    plt.figure(figsize=(20,10))
    orig = plt.plot (bldg_specific_steam, color = 'blue',label = (BLDG))
    mean = plt.plot (rolmean, color = 'red',label = 'Rolling Mean')
    std = plt.plot (rolstd, color = 'black',label = 'Rolling std')
    plt.legend (loc ='best')
    plt.title ('Rolling Mean & Standard Deviation')
    plt.show (block = False)
print(bldg_specific_steam)


# ## Automatic Time Series Decomposition

# In[ ]:



stm_df = stm_df.fillna(4)

for time_series in all_buildings:
    bldg_specific_steam = stm_df[time_series]
    bldg_specific_steam= pd.DataFrame(bldg_specific_steam)
    decomposition = seasonal_decompose (bldg_specific_steam.values,period = 24*30, model = 'additive') 
    decomposition.plot()
    plt.title(time_series)
    plt.tight_layout()

print('The result:')
print(decomposition.observed)
print(decomposition.trend)
print(decomposition.seasonal)
print(decomposition.resid)



# ## Build ARIMA

# ## Determine the order of AR, I and MA component 
# AR = p = period for autoregressive model (regression the past lag value, ACF method),
# <br>
# Integrated = d = order of autoregression (differenced value from present and previous to eliminate the effects of seasonality; removing the trend and seasonality to make it stationary)
# <br>
# MA = q = periods in moving average (present value is not only depended on the past value but the error lag value as well, use the ACF method)
# <br>
# Using PAFC autocorreclation plot and PACF partial autocorrelatioin plot

# In[ ]:


stm_df = stm_df.fillna(4)
for BLDG in all_buildings:
 
    bldg_specific_steam = stm_df[BLDG]
    bldg_specific_steam= pd.DataFrame(bldg_specific_steam)
    #print(bldg_specific_steam)
    size = int(len(bldg_specific_steam) * 0.5)
    train, test = bldg_specific_steam[0:size], bldg_specific_steam[size:len(bldg_specific_steam)]
    model = ARIMA(train, order=(3,0,5))
    results_ARIMA = model.fit()
    predictions = results_ARIMA.predict(start = len (train), end = len(bldg_specific_steam)+24*7, typ = 'levels'). rename ('ARIMA predictions')
    #print(predictions)
    #pd.DataFrame(pred)
    mean_value =   bldg_specific_steam.mean()
    rmse = sqrt(mean_squared_error(test, predictions))
    RMSE_mean = (rmse//mean_value)
 
    print('Mean of usage: ', (mean_value))
    print('Test RMSE: ', (rmse))
    print('Test RMSE/mean: ', (RMSE_mean))
    print('predicted and building', (predictions, BLDG))
    
print('Test RMSE: %.3f' % rmse)


# In[ ]:





# In[ ]:





# In[ ]:




