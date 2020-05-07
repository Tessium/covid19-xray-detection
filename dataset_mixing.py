#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import os


# In[2]:


df = pd.read_csv('meta.csv')  


# In[ ]:





# In[15]:


for index, row in df.iterrows():
    if row['Label_1_Virus_category'] == 'Virus':
        name = row['X_ray_image_name']
        try:
            os.rename('datasets/' + row['Dataset_type'].lower() + '/' + name, 
                      'datasets/virus/' + name)
        except:
            pass


# In[ ]:




