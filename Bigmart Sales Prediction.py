#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[3]:


df = pd.read_csv("Downloads/Data-Analytics-Projects-master/Bigmart Sales Prediction/train.csv")


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.nunique()


# In[8]:


df.isnull()


# In[9]:


# df.isnull
# df.isna()


# In[10]:


df1 = pd.read_csv("Downloads/Data-Analytics-Projects-master/Bigmart Sales Prediction/test.csv")


# In[11]:


df1.head(3)


# In[15]:


train_x = df.drop(columns = ['Item_Outlet_Sales'], axis = 1)
train_y = df['Item_Outlet_Sales']


# In[16]:


test_x = df1.drop(columns = ['Item_Outlet_Sales'], axis = 1)
test_y = df1['Item_Outlet_Sales']


# In[17]:


lin_model = LinearRegression()


# In[18]:


lin_model.fit(train_x, train_y)


# In[21]:


lin_model.coef_


# In[23]:


lin_model.intercept_


# In[24]:


predict_train = lin_model.predict(train_x)


# In[25]:


predict_train


# In[29]:


rmse_train = mean_squared_error(train_y, predict_train)**0.05


# In[30]:


rmse_train


# In[32]:


predict_test = lin_model.predict(test_x)


# In[33]:


predict_test


# In[34]:


rmse_test = mean_squared_error(test_y, predict_test)**0.05


# In[35]:


rmse_test

