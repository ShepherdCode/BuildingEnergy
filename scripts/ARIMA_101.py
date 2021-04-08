#!/usr/bin/env python
# coding: utf-8

# # Apply ARIMA to Weather Data
# Assume user downloaded archive.zip from Kaggle,
# renamed the file BuildingData.zip,
# and stored the file in the data subdirectory.
# Assume the zip file contains the weather.csv file.  

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
WEATHER_FILE='weather.csv'
MODEL_FILE='Model'  # will be used later to save models


# In[2]:


from os import listdir
import csv
from zipfile import ZipFile
import numpy as np
import pandas as pd
from pandas.plotting import autocorrelation_plot

from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.arima.model import ARIMA

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
    temp_col = pd.to_datetime(panda['timestamp'])
    temp_tab = panda.drop(['timestamp'],axis=1)
    panda = temp_tab
    panda.insert(0,'date',temp_col,True)
    return panda
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


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)
print(wet_df.info())
print("Index:",wet_df.index)
# This dataframe has 331K rows and occupies 25MB RAM.
# The date column contains duplicates across different sites.


# In[5]:


site_df = get_site_timeseries(wet_df,'Gator')
print(site_df.info())
print("Index:",site_df.index)
# This dataframe represents just one site.
# This dataframe 18K rows and occupies 1MB RAM.
# Since the dates are unique, converted that column to the index.


# ## Weather data
# We have 2 years of hourly weather data per site ID.  
# A site is a geographical area such as a college campus.  
# Each site is code-named with an animal like Bear.
# For each site, we have multiple buildings.  
# Each building is code-named with person-name like Lulu.  

# ## Plot temperature
# Note two-year cycle.

# In[6]:


sites=wet_df.site_id.unique()


# In[7]:


# Plot temperature time series per site. 
# Separate plots stacked vertically.
for site in sites:
    site_df = get_site_timeseries(wet_df,site)
    temp_df = site_df['airTemperature']
    temp_df.plot(figsize=(20,5))
    plt.title(site)
    plt.show()


# In[8]:


# Plot temperature time series per site. 
# Plot all the sites against the date axis.
sites=wet_df.site_id.unique()
for site in sites:
    site_df = get_site_timeseries(wet_df,site)
    temp_df = site_df['airTemperature']
    temp_df.plot(figsize=(20,10))
print("At every site, air temp shows a 2-year cycle that peaks in summer.")
plt.show()


# ## Downsampling
# For example, replace every 24 hourly time steps with one daily average.

# In[9]:


site='Gator'
site_df = get_site_timeseries(wet_df,site)
temp_df = site_df['airTemperature']
smooth = temp_df.resample("24H").mean()
smooth.plot(figsize=(20,10))
plt.title("Site "+site+" air temp downsampled using daily mean")
plt.show()


# In[10]:


for site in sites:
    site_df = get_site_timeseries(wet_df,site)
    temp_df = site_df['airTemperature']
    smooth = temp_df.resample("7D").mean()
    smooth.plot(figsize=(20,10))
plt.title("All sites air temp downsampled using weekly mean")
plt.show()


# ## Smoothing
# For example, replace every hourly time step with the 24H window average.

# In[11]:


site='Gator'
site_df = get_site_timeseries(wet_df,site)
temp_df = site_df['airTemperature']
smooth=temp_df.rolling(window=24).mean() # first n-1 get set to NaN
smooth.plot(figsize=(20,10))
plt.title("Site "+site+" air temp smoothed using daily mean")
plt.show()


# In[12]:


for site in sites:
    site_df = get_site_timeseries(wet_df,site)
    temp_df = site_df['airTemperature']
    smooth=temp_df.rolling(window=24*7).mean() # first n-1 get set to NaN
    smooth.plot(figsize=(20,10))
plt.title("All sites air temp smoothed using weekly mean")
plt.show()


# ## Autocorrelation

# In[13]:


site='Gator'
site_df = get_site_timeseries(wet_df,site)
variable='airTemperature'
series = site_df[variable]
print(type(series))
series.head()


# In[14]:


days=12*24
print("Analysis of",variable,"at the",site,"site.")
print("Autocorrelation measured over first",days,"days.")
print("X-axis=hours. Note 24-hour lag starts stong then dissipates.")
plt.figure(figsize=(20,5));
major_ticks = np.arange(0, days, 24)  # grid marks on the 12-hour, 24-hour boundary
minor_ticks = np.arange(0, days, 12)  # grid marks on the 12-hour, 24-hour boundary
# fig, ax = plt.subplots()
ac_axis = autocorrelation_plot(series[:days])
ac_axis.set_xticks(major_ticks,minor=False)
ac_axis.set_xticks(minor_ticks,minor=True)
plt.show()


# ## ARIMA

# In[15]:


site='Gator'
site_df = get_site_timeseries(wet_df,site)
variable='airTemperature'
series = site_df[variable]

p = 15  # AR = lag order e.g. number of days to look back
d = 0  # I = degree of differncing (use zero if not stationary)
q = 24 # MA = order of moving average

if False:   # This works but it takes a long time (10 min?)
    arima= ARIMA(series, order=(p,d,q)).fit()
    print(arima.summary())
    residuals = pd.DataFrame(arima.resid)
    residuals.plot()
    plt.show()
    # density plot of residuals
    residuals.plot(kind='kde')
    plt.show()
    # summary stats of residuals
    print(residuals.describe())


# In[ ]:





# ## Useful references
# 
# statsmodels ARIMA  
# https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARIMA.html
# 
# Jason Brownlee  
# https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
# 
# Kaggle (covers a lot but explains a little)  
# https://www.kaggle.com/sumi25/understand-arima-and-tune-p-d-q

# In[ ]:




