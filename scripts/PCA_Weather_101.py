#!/usr/bin/env python
# coding: utf-8

# ## PCA weather

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
WEATHER_FILE='weather.csv'
MODEL_FILE='Model'  # will be used later to save models


# In[ ]:


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


wet_df = read_zip_to_panda(ZIP_PATH,WEATHER_FILE)
wet_df = fix_date_type(wet_df)


# In[ ]:


wet_df = wet_df.loc[:,'airTemperature': 'windSpeed'] #Create a new dataframe not including timestamp and site id


# In[ ]:


#Fit and transform in standard scaler 
scaler = StandardScaler ()
scaler.fit(wet_df)
wet_df_transform = scaler.transform (wet_df) #Apply transform 


# In[ ]:


#Input contains NaN, infinity or a value too large for dtype('float64').
pca_wet_df = pd.DataFrame(wet_df_transform) #create dataframe
np.where(pca_wet_df .values >= np.finfo(np.float64).max) 
pca_wet_df.replace([np.inf, -np.inf], np.nan, inplace=True)
pca_wet_df.fillna(0, inplace=True)
pca_wet_df = pca_wet_df.to_numpy() #convert Dataframe back to array

pca = PCA (0.98) #shows 98% of the data in PCA
pca.fit (pca_wet_df)
pca_wet_df = pca.transform (pca_wet_df)

per_var = np.round(pca.explained_variance_ratio_*100,decimals = 1)
labels = ['PC'+str(x) for x in range (1,len(per_var)+1)]

plt.figure(figsize=(10,7))
plt.bar(x=range(1,len(per_var)+1), height = per_var, tick_label =labels)
plt.ylabel('Percentage of Variance',fontsize = 20)
plt.xlabel('Principal component',fontsize = 20)
plt.title('PCA Weather',fontsize = 25)
plt.show()


# In[ ]:





# In[ ]:




