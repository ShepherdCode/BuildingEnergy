#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing the libraries
import numpy as np
import pandas as pd
from numpy import mean
from numpy import std
from pandas import DataFrame


# In[36]:



#importing data as csv
dataset = pd.read_csv('C:/Users/Owner/Desktop/weather.csv')
#print(type(dataset))


# In[37]:


dataset


# In[40]:


#Air Temperature for panther
air_temperature= dataset['airTemperature']
print(air_temperature)
air_temperature_panther = air_temperature.loc[0:17545]
air_temperature_panther
#Average for Panther Air Temperature
ave_panther_airTemp = air_temperature_panther.mean(axis=0)
print(ave_panther_airTemp)


# In[41]:


#Dataset for panther
set_panther = dataset.loc[0:17545,'airTemperature':'windSpeed']


# In[42]:


set_panther


# In[43]:


#Average dataset for panther
avg_panther = set_panther.mean(axis=0)
avg_panther


# Dataset for Robin

# In[44]:


set_robin = dataset.loc[17546:35061,'airTemperature':'windSpeed']
set_robin


# In[45]:


#Average dataset for robin
avg_robin = set_robin.mean(axis=0)
avg_robin


# Dataset for Fox

# In[46]:


set_fox = dataset.loc[35062:52604,'airTemperature':'windSpeed']
set_fox


# In[47]:


#Average dataset for fox
avg_fox = set_fox.mean(axis=0)
avg_fox 


# Dataset for Rat

# In[48]:


set_rat= dataset.loc[52605:70143,'airTemperature':'windSpeed']
set_rat


# In[49]:


#Average dataset for rat
avg_rat = set_rat.mean(axis=0)
avg_rat


# Dataset for Bear

# In[50]:


set_bear= dataset.loc[70144:87685,'airTemperature':'windSpeed']
set_bear


# In[51]:


#Average dataset for bear
avg_bear = set_bear.mean(axis=0)
avg_bear


# Dataset for Lamb

# In[52]:


set_lamb= dataset.loc[87686:105185,'airTemperature':'windSpeed']
set_lamb


# In[53]:


#Average dataset for lamb
avg_lamb = set_lamb.mean(axis=0)
avg_lamb


# Dataset for peacock

# In[54]:


set_peacock= dataset.loc[105186:122724,'airTemperature':'windSpeed']
set_peacock


# In[55]:


#Average dataset for peacock
avg_peacock = set_peacock.mean(axis=0)
avg_peacock


# Dataset for Moose

# In[56]:


set_moose= dataset.loc[122725:139584,'airTemperature':'windSpeed']
set_moose


# In[57]:


#Average dataset for moose
avg_moose = set_moose.mean(axis=0)
avg_moose


# Dataset for Gator

# In[58]:


set_gator= dataset.loc[139585:157127,'airTemperature':'windSpeed']
set_gator


# In[59]:


#Average dataset for gator
avg_gator = set_gator.mean(axis=0)
avg_gator


# Dataset for bull

# In[60]:


set_bull= dataset.loc[157128:174657,'airTemperature':'windSpeed']
set_bull


# In[61]:


#Average dataset for bull
avg_bull = set_bull.mean(axis=0)
avg_bull


# Dataset for Bobcat

# In[62]:


set_bobcat= dataset.loc[174658:192182,'airTemperature':'windSpeed']
set_bobcat


# In[63]:


#Average dataset for bobcat
avg_bobcat= set_bobcat.mean(axis=0)
avg_bobcat


# Dataset for Crow

# In[64]:


set_crow= dataset.loc[192183:209042,'airTemperature':'windSpeed']
set_crow


# In[65]:


#Average dataset for bobcat
avg_crow= set_crow.mean(axis=0)
avg_crow


# Dataset for shrew

# In[66]:


set_shrew= dataset.loc[209043:226558,'airTemperature':'windSpeed']
set_shrew


# In[67]:


#Average dataset for shrew
avg_shrew= set_shrew.mean(axis=0)
avg_shrew


# Dataset for swan

# In[68]:


set_swan= dataset.loc[226559:244093,'airTemperature':'windSpeed']
set_swan


# In[69]:


#Average dataset for shrew
avg_swan= set_swan.mean(axis=0)
avg_swan


# Dataset for wolf

# In[70]:


set_wolf= dataset.loc[244094:261598,'airTemperature':'windSpeed']
set_wolf


# In[71]:


#Average dataset for wolf
avg_wolf= set_wolf.mean(axis=0)
avg_wolf


# Dataset for Hog

# In[72]:


set_hog= dataset.loc[261599:279140,'airTemperature':'windSpeed']
set_hog


# In[73]:


#Average dataset for hog
avg_hog= set_hog.mean(axis=0)
avg_hog


# Dataset for Eagle

# In[74]:


set_eagle= dataset.loc[279141:296676,'airTemperature':'windSpeed']
set_eagle


# In[75]:


#Average dataset for eagle
avg_eagle= set_eagle.mean(axis=0)
avg_eagle


# Dataset for cockatoo

# In[76]:


set_cockatoo= dataset.loc[296677:313651,'airTemperature':'windSpeed']
set_cockatoo


# In[77]:


#Average dataset for cockatoo
avg_cockatoo= set_cockatoo.mean(axis=0)
avg_cockatoo


# Dataset for mouse

# In[78]:


set_mouse= dataset.loc[313652:331167,'airTemperature':'windSpeed']
set_mouse


# In[79]:


#Average dataset for mouse
avg_mouse= set_mouse.mean(axis=0)
avg_mouse


# In[80]:


avg_mouse_dataframe = pd.DataFrame(data= [avg_panther,avg_robin,avg_rat,avg_panther,avg_panther,avg_panther], 
                                   columns = ['Panther','Robin','Fox','Rat',
                                            'Bear','Lamb','Peacock','Moose','Gator','Bull',
                                            'Bobcat','Crow','Shrew','Swan','Wolf','Hog','Eagle','Cockatoo',
                                            'Mouse'])
avg_mouse_dataframe 


# In[81]:


avg_mouse_dataframe = pd.DataFrame(data= avg_panther, columns = ['panther'])
avg_mouse_dataframe 


# In[82]:


#average_dataframe=pd.DataFrame(data= 'airTemperature','cloudCoverage',  columns = ['panther','robin'])
#average_dataframe.head(2)


# In[ ]:





# In[ ]:




