#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

# Encode to one_hot format
def encode(class_name, classes):

    one_hot = np.zeros(shape=(len(classes)), dtype=np.int8)
    class_index = classes.index(class_name)
    one_hot[class_index] = 1

    return one_hot

# Decode from one_hot format to string
def decode(index, classes):
    
    return classes[index]


# In[ ]:




