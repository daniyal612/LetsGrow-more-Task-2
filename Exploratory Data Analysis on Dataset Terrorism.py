#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import plot_tree


# In[2]:


#Importing Data
data=pd.read_csv("E:\Iris.csv")


# In[3]:


# Data understanding and preparation
data.head()


# In[4]:


data.shape


# In[5]:


data.columns


# In[6]:


data=data.drop(['Id'],axis=1)


# In[7]:


x=data.drop(['Species'],axis=1)
y=data['Species']


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[9]:


# Model building
clf=DecisionTreeClassifier()
clf.fit(x_train,y_train)


# In[10]:


# Predicting on test data
predict=clf.predict(x_test)


# In[11]:


predict


# In[12]:


confusion_matrix(y_test,predict)


# In[13]:


# Visualization of the Tree
feature_names = data.columns[:4]
target_names = data['Species'].unique().tolist()
plt.figure(figsize=(20,15))
a=plot_tree(clf,
           feature_names = feature_names,
           class_names = target_names,
           filled = True,
           rounded = True)


# In[ ]:




