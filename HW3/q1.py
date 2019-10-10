#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from googlesearch import get_freq


# In[2]:


N = 25270000000000 #1000*(no of webpages for search 'a')
def get_NGD(w1, w2):   
    fx = get_freq(w1)
    fy = get_freq(w2)
    fxy = get_freq(w1 + '+' + w2)
    NGD = (max(np.log(fx), np.log(fy)) - np.log(fxy))/(np.log(N) - min(np.log(fx), np.log(fy)))
    
    return NGD


# In[3]:


w1 = input('Enter Word 1: ')
w2 = input('Enter Word 2: ')


# In[4]:


NGD = get_NGD(w1, w2)
print(NGD)

