#!/usr/bin/env python
# coding: utf-8

# In[3]:


from googlesearch import get_freq
import numpy as np
import pandas as pd
import time


# In[4]:


data = pd.read_csv('data.csv', delimiter=',')


# In[5]:


x = data['Word 1'].values
y = data['Word 2'].values
human_dist = data['Human (mean)'].values


# In[6]:


freq_data = np.zeros((x.shape[0],4))


# In[7]:


tick = time.time()
for i in range(x.shape[0]):
    print('No: ', i, ' | Time: ', time.time() - tick)
    
    freq_data[i, 0] = get_freq(x[i])
    time.sleep(0.1)
    
    freq_data[i, 1] = get_freq(y[i])
    time.sleep(0.1)
    
    freq_data[i, 2] = get_freq(x[i] + '+' + y[i])
    time.sleep(0.1)
    
    freq_data[i, 3] = human_dist[i]


# In[33]:


np.savetxt('freq_data.csv', freq_data, delimiter=',')

