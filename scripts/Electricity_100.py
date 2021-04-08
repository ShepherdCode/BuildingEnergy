#!/usr/bin/env python
# coding: utf-8

# # Electricity Dataset
# Assume user downloaded archive.zip from Kaggle,
# renamed the file BuildingData.zip,
# and stored the file in the data subdirectory.
# Assume the zip file contains the electricity_cleaned.csv file.  

# In[14]:


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


# In[15]:


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


# In[16]:


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


# In[17]:


elec_df = read_zip_to_panda(ZIP_PATH,ELEC_FILE)
elec_df


# In[18]:


def show_sample_panda(df):
    print("The data is stored in this class:",type(df))
    print("Shape:",df.shape)
    data=df.iloc[0]
    print("First two column names:",df.columns[0],df.columns[1])
    print("First two column types:",df.dtypes[0],df.dtypes[1])
    print("First data row:",data[0],data[1],"...",data[-2],data[-1])
    data=df.iloc[-1]
    print("Last two column names:",df.columns[-2],df.columns[-1])
    print("Last two column types:",df.dtypes[-2],df.dtypes[-1])
    print("Last data row:",data[0],data[1],"...",data[-2],data[-1])
    df.describe()
show_sample_panda(elec_df)


# In[19]:


# Pandas statistics per column. Counts excludes NaN columns and values.
elec_df.describe()


# In[20]:


park_cols = [c for c in elec_df.columns if 'parking' in c]
print(park_cols)


# In[22]:


elec_df.boxplot(park_cols,rot=90)


# In[13]:


def describe_columns(df,cols):
    data_source = {}
    usage_category = {}
    building = {}
    for c in df.columns:
        if 'timestamp' not in c:
            w1,w2,w3=c.split('_')        
            data_source[w1] = data_source.get(w1,0)+1
            usage_category[w2] = usage_category.get(w2,0)+1
            building[w3] = building.get(w3,0)+1
    print("Data sources are given animal code names like Bear.")
    print("Data sources are single locations e.g. a college campus.")
    print("We have one stream of weather data from each data source.")
    print("Number of data sources:",len(data_source))
    print("Number of buildings with data per data source:\n",data_source)
    print()
    print("Buildings are categorized by primary space usage like Parking.")
    print("Number of data categories:",len(usage_category))
    print("Number of buildings with data per category:\n",usage_category)
    print()
    print("Each building is given a person name like Lula.")
    print("Number of distinct building names:",len(building))
    print()
    print("Note that Robin is an animal, not a building name.")
    
describe_columns(elec_df,elec_df.columns)


# In[ ]:





# In[ ]:




