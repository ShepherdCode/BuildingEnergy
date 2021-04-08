#!/usr/bin/env python
# coding: utf-8

# # Apply ARIMA to the Electricity Dataset
# Assume user downloaded archive.zip from Kaggle,
# renamed the file BuildingData.zip,
# and stored the file in the data subdirectory.
# Assume the zip file contains the electricity_cleaned.csv file.  

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
ELEC_FILE='electricity_cleaned.csv'
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

from statsmodels.tsa.arima.model import ARIMA

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
def get_building_timeseries(panda,building):
    # Assume the panda dataframe has a datetime column.
    # (If not, call fix_date_type() before this.)
    # Extract the timeseries for one site.
    # Convert the datetime column to a DatetimeIndex.
    site_df = panda[panda['site_id']==site]
    temp_col = site_df['date']
    temp_val = temp_col.values
    temp_ndx = pd.DatetimeIndex(temp_val)
    dropped = building_df.drop('date',axis=1)
    panda = dropped.set_index(temp_ndx)
    return panda


# In[ ]:


elec_df = read_zip_to_panda(ZIP_PATH,ELEC_FILE)
elec_df = fix_date_type(elec_df)
elec_df.info()


# In[ ]:





# In[ ]:


#Convert the DataTime
#elec_df['timestamp'] = pd.to_datetime(elec_df['timestamp'], infer_datetime_format = True)
#indexed_elec_df = elec_df.set_index(['timestamp'])


# In[ ]:





# In[ ]:


park_cols = [c for c in elec_df.columns if 'Gator' in c]
print(park_cols)


# In[ ]:


cols=elec_df.columns
all_sites=[s.split('_')[0] for s in cols]  # Site is first part of building name like Hog_parking_Linda
uniq_sites = [x for x in set(all_sites)]
site_counts = [[x,all_sites.count(x)] for x in set(all_sites)]
print("Note we only have a few sites!")
print("Buildings per site:\n",site_counts)


# In[ ]:


buildings = elec_df.filter(like='Fox')


# In[ ]:


list_buildings= list(elec_df)


# In[ ]:


# Plot temperature time series for one column (building). 
show_all_plots = False
if show_all_plots:
    cols=elec_df.columns
    uniq_sites = [x for x in set(all_sites)]
    for site in uniq_sites:
        for bldg in cols:
            if bldg.startswith(site):
                temp_df = stm_df[bldg]
                temp_df.plot(figsize=(20,5))
        plt.title("Steam for site "+site)
        plt.show()


# In[ ]:





# ## Drill Down
# Number of buildings for Bull = 4; Moose = 6; Rat = 7; Eagle = 19; Wolf = 33; Bear = 34; Peacock = 35; Robin = 50; Lamb = 77; Fox and Hog= 116. 

# In[ ]:


# Plot temperature time series for one column (building). 
cols=elec_df.columns
uniq_sites = [x for x in set(all_sites)]
show_legend = False
for site in uniq_sites:
    suitable_bldgs=[]
    for bldg in cols:
        if bldg.startswith(site):  # and 'education' in bldg and bldg.endswith('ll'):
            temp_df = elec_df[bldg]
            smooth = temp_df.resample("1D").mean()  
            count_nan=np.isnan(smooth).sum()
            count_zero=smooth.isin([0]).sum()
            if count_nan<=0 and count_zero<=5:
                suitable_bldgs.append(bldg)
                np.seterr(divide = 'ignore') 
                logs = np.log(smooth)
                logs.plot(figsize=(20,10))
                np.seterr(divide = 'warn') 
    num_bldg = len(suitable_bldgs)
    if num_bldg>0:
        plt.title("Electricity for selected buildings at site "+site)
        if show_legend:
            plt.legend()
        plt.show()
        print("Site",site,"has good steam data from",num_bldg,"buildings:\n",suitable_bldgs)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# See the Electricity_100 notebook for basic stats.


# In[ ]:


# Not done: start ARIMA analysis.

