#!/usr/bin/env python
# coding: utf-8

# In[14]:


import os
import re
import random
import argparse
import numpy as np
from PIL import Image
import csv
import imageio
from extract import Extractor
from one_hot import encode
# to save the png format of all the training and testing data.
# from extract import save_data


# In[15]:


def get_data():

    extractor = Extractor(32, "2012")
    train_data, test_data, test_data_char = extractor.pixels()

    train_inputs = []
    train_labels = []

    test_inputs = []
    test_labels = []

    test_inputs_char = []
    test_labels_char = []

    #extracting the training data
    for i in train_data:
        train_inputs.append(i['features'])
        train_labels.append(i['label'])
        
    train_inputs = np.array(train_inputs)
    train_inputs = np.reshape(train_inputs, (-1, 1, 32 ,32))
    train_inputs = np.transpose(train_inputs, axes=[0,2,3,1])

    train_labels = [encode(train_label, extractor.classes) for train_label in train_labels]
    train_labels = np.asarray(train_labels)

    #extracting the testing data to make the dataset of full mathematical expressions

    for i in test_data:
        test_input = []
        test_label = []

        for j in i:
            test_input.append(j['features'])
            test_label.append(j['label'])
        
        test_label = np.array(test_label)
        test_label = [encode(test_i, extractor.classes) for test_i in test_label]

        test_inputs.append(test_input)
        test_labels.append(test_label)

    #extracting testing data to make a dataset of individual mathematical symbols
    for i in test_data_char:
        test_inputs_char.append(i['features'])
        test_labels_char.append(i['label'])

    test_inputs_char = np.array(test_inputs_char)
    test_inputs_char = np.expand_dims(test_inputs_char, 0)
    test_inputs_char = np.reshape(test_inputs_char, (-1,32,32,1))
    test_labels_char = [encode(test_label_char, extractor.classes) for test_label_char in test_labels_char]
    test_labels_char = np.array(test_labels_char)

    return train_inputs, train_labels, test_inputs, test_labels, test_inputs_char, test_labels_char, extractor
    


# In[ ]:





# In[ ]:




