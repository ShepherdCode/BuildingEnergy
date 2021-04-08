#!/usr/bin/env python
# coding: utf-8

# # Meta Data
# Assume user downloaded archive.zip from Kaggle,
# renamed the file BuildingData.zip,
# and stored the file in the data subdirectory.
# Assume the zip file contains the metadata.csv file.  

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
META_FILE='metadata.csv'
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


# In[4]:


meta_df = read_zip_to_panda(ZIP_PATH,META_FILE)
meta_df


# In[5]:


print("Columns:\n",meta_df.columns)
print("Shape:",meta_df.shape)
NUM_ROWS=meta_df.shape[0]
NUM_COLS=meta_df.shape[1]


# In[6]:


# Pandas statistics per column. Counts excludes NaN columns and values.
meta_df.describe()


# In[7]:


print("For each column, number of rows in which column is not null:")
print(meta_df.count(axis=0))


# ## Site
# Site corresponds to a single data source such as a college campus.  
# Each site has an animal code name like Bear.  

# In[8]:


meta_df.site_id.value_counts()


# ## Meter Type
# Meter types include solar, electric, hot water.  
# If a building says electricity=yes, then we have electric meter data for that building.  
# Some buildings have none, some have several.  

# In[9]:


meter_types=['electricity','hotwater','chilledwater','steam','irrigation','solar','gas']
meta_df.groupby('site_id').count()[meter_types]


# In[10]:


print("Building count for each meter type (some buildings have several):")
meta_df[meter_types].count()


# In[11]:


print("All the solar buildings come from one data source (bobcat):")
for i in range(0,NUM_ROWS):
    row=meta_df.iloc[i]
    bldg=row['building_id']
    solar=row['solar']
    if pd.notnull(solar):
        print(bldg,solar)


# In[13]:


print("Warning: meter type is not yes/no, it is yes/nan.")
print("Nan (not-a-number) must be handled specially e.g. the isna() method.")
for col in meter_types:
    print(col,meta_df[col].unique())


# In[14]:


print("Buildings can many energy meters. Example:")
for i in range(0,NUM_ROWS):
    row=meta_df.iloc[i]
    bldg=row['building_id']
    cnt = 0
    if row['electricity']=='Yes':
        cnt += 1
    if row['hotwater']=='Yes':
        cnt += 1
    if cnt>1:
        print('Electricity and hot water meters:',bldg)
        break
        


# ## Energy Ratings
# We had hoped to predict energy ratings from other features.  
# Unfortunately, the number of rated buildings is under 200.

# In[15]:


print("Energy rating columns contain these values")
energy_types=['energystarscore','leed_level','rating']
for col in energy_types:
    print(col,':\n',meta_df[col].unique())


# In[16]:


meta_df[energy_types].count()


# In[17]:


print("Value counts:")
print(meta_df.rating.value_counts(dropna=False))


# In[18]:


print("Value counts:")
print(meta_df.leed_level.value_counts(dropna=False))


# In[19]:


print("Value counts:")
print(meta_df.energystarscore.value_counts(dropna=False))


# ## EUI
# EUI is the miles-per-gallon for a building. Some entries are non-numeric.  
# We could calculate it from energy usage per square foot?  
# We have 300 buildings with given EUI.  
# Most values are nan. A few values are "-".

# In[20]:


energy_types=['eui','site_eui','source_eui']
for col in energy_types:
    print(col)
    print(" count nan:",meta_df[col].isna().sum()," count -:",meta_df[meta_df[col]=='-'].shape[0])


# In[21]:


meta_df[['eui','site_eui','source_eui']].count()


# ## Building age
# We could use age as a classification to be predicted.  
# Example: binary, pre-1950 vs post-1950.  
# Potential problem: old buildings might have been refurbished.

# In[22]:


print("Count buildings with known year built:", meta_df.yearbuilt.count())


# In[23]:


print("Count buildings binned by year built:")
meta_df.yearbuilt.value_counts(bins=10)


# In[24]:


meta_df.hist(['yearbuilt'])


# ## Other fields

# In[25]:


print("Heating Type (most are gas)")
print(meta_df.heatingtype.unique())
meta_df.heatingtype.value_counts(dropna=False)


# In[26]:


print("Number of Floors (most are one or two)")
meta_df.numberoffloors.value_counts(dropna=False)


# In[27]:


print("Occupants (the mode is 100)")
meta_df.occupants.count()


# In[28]:


meta_df.occupants.hist()


# In[29]:


print("Industry")
meta_df.industry.value_counts(dropna=False)


# In[30]:


print("Time Zone")
meta_df.timezone.value_counts(dropna=False)


# In[32]:


print("Square Meters")
meta_df.sqm.hist(log=True)


# In[31]:


print("Latitude")
meta_df.lat.hist()


# In[36]:


geo_fields=['lat','lng']
meta_df.groupby('site_id').count()[geo_fields]


# In[ ]:





# In[ ]:




