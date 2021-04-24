#!/usr/bin/env python
# coding: utf-8

# # Weather Data
# Assume user downloaded archive.zip from Kaggle,
# renamed the file BuildingData.zip,
# and stored the file in the data subdirectory.
# Assume the zip file contains the weather.csv file. 
# 
# The weather file has one row per hour for two years with 8 feature columns. We noted a large range of mean air temperature per site: from 7.8 to 25.1 degrees Celsius.

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
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import colors
mycmap = colors.ListedColormap(['red','blue'])  # list color for label 0 then 1
np.set_printoptions(precision=2)


# In[3]:


def read_csv_to_numpy(filename): # array of string, header=row[0]
    with open(ELEC_PATH,'r') as handle:
        data_iter = csv.reader(handle,delimiter = ',',quotechar = '"')
        data = [data for data in data_iter]
        return np.asarray(data, dtype = None)
# Pandas incorporates column headers, row numbers, timestamps, and NaN for missing values.
def read_csv_to_panda(filename): # pandas data frame
    return pd.read_csv(filename)
def read_zip_to_panda(zip_filename,csv_filename):
    zip_handle = ZipFile(zip_filename)
    csv_handle = zip_handle.open(csv_filename)
    panda = pd.read_csv(csv_handle)
    return panda


# ## Weather data
# We have 2 years of hourly weather data per site ID.  
# A site is a geographical area such as a college campus.  
# Each site is code-named with an animal like Bear.
# For each site, we have multiple buildings.  
# Each building is code-named with person-name like Lulu.  

# In[4]:


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)


# In[5]:


wet_df


# In[6]:


print("Air temp and wind speed: 300K reports.")
wet_df.describe()


# In[7]:


print("Weather observations per site:")
wet_df.site_id.value_counts()


# In[8]:


print("Outside Air Temperature observations for one site:")
gator_df = wet_df[wet_df['site_id']=='Gator']
gator_temp_df=gator_df['airTemperature']
gator_temp_df.describe()


# In[9]:


print("Stats for outside air temp per site:")
wet_df.groupby(by=['site_id'])['airTemperature'].agg(['mean','std','min','max']).sort_values('mean')

