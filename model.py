#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np
import os


# In[2]:


df = pd.read_csv(".//iris.csv")
df


# In[3]:


X = df.drop(columns=['species'],axis=1)
Y = df['species']


# In[4]:


from sklearn.ensemble import GradientBoostingClassifier
model = GradientBoostingClassifier()


# In[5]:


model.fit(X,Y)


# In[ ]:




