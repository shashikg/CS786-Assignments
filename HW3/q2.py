#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


N = 25270000000000 #1000*(no of webpages for search 'a')
def calc_NGD(fx, fy, fxy):   
    NGD = (max(np.log(fx), np.log(fy)) - np.log(fxy))/(np.log(N) - min(np.log(fx), np.log(fy)))
    return NGD


# In[3]:


data = np.genfromtxt('freq_data.csv',delimiter=',')


# In[4]:


NGD = np.zeros(data.shape[0])


# In[5]:


for i in range(data.shape[0]):
    NGD[i] = calc_NGD(data[i, 0], data[i, 1], data[i, 2])


# In[6]:


NGD_Scaled = max(NGD) - NGD
NGD_Scaled = 10*(NGD_Scaled/max(NGD_Scaled))


# In[7]:


plt.axis('equal')
plt.plot(data[:,3], NGD_Scaled, '*')
plt.title('Scaled NGD vs Human Similarity')
plt.ylabel('Scaled NGD')
plt.xlabel('Human Similarity')
plt.axis([0, 10, 0, 10])
plt.savefig("NGDvsHumanSimilarity.png", dpi = 200)

