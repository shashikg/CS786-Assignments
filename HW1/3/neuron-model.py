#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def fx(x, y, alpha, sigma):
    if x<=0:
        return (y + alpha/(1-x))
    elif x >= (alpha + y):
        return -1
    else:
        return (alpha + y)


# In[3]:


def fy(x, y, alpha, sigma):
    return (y - mu*(x + 1) + mu*sigma)


# In[4]:


mu = 0.001
alphas = np.linspace(2, 8, 5, endpoint=True)
sigmas = np.linspace(-1, 1, 20, endpoint=True)

N = 2000


# In[5]:


for m in range(sigmas.shape[0]):
    for n in range(alphas.shape[0]):
        x = np.zeros(N)
        y = np.zeros(N)
        y[0] = 1 - alphas[n] #this gives a better start to observe y and x
        
        for i in range(N-1):
            x[i + 1] = fx(x[i], y[i], alphas[n], sigmas[m])
            y[i + 1] = fy(x[i], y[i], alphas[n], sigmas[m])
            
        print(m, n)    
        plt.plot(x,  linewidth=0.3)
        plt.ylabel('x')
        plt.xlabel('n')
        plt.title("alpha: " + str(alphas[n]) + ", sigma: " + str(sigmas[m]))
        plt.savefig("outputs/" + str(m) + str(n) + ".png", dpi = 200)
        plt.close()


# In[ ]:




