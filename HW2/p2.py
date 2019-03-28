#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from SearchParadigm import get_paradigm


# In[2]:


print("Make sure to enter N>1 for  conjunction search!")

N = int(input("No of Objects: "))
E = int(input("Experiment Type [0 - Feature; 1 - Conjunction]: "))

object_location, paradigm_img = get_paradigm(N, E)
    
io.imshow(paradigm_img)
plt.show()


