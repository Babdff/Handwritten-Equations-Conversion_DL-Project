#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import
from matplotlib import pyplot as plt
from numpy.core.arrayprint import format_float_positional
from numpy.lib.function_base import _DIMENSION_NAME, select
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.ops.gen_nn_ops import MaxPool
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D, Dropout
import numpy as np


# In[2]:


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class contains the architecture for your CNN that 
        classifies mathematical symbols. 
        """
        super(Model, self).__init__()

        self.num_classes = 101
        self.batch_size = 250
        self.num_epochs = 5
        self.hidden_dim = 500

        # Append losses to this list in training so you can visualize loss vs time in main
        self.loss_list = [] 
    
        #optimizer
        self.optimization = tf.keras.optimizers.Adam(learning_rate=1e-3)

        self.architecture = [
                Conv2D(32,5,1,padding="same",
                   activation="relu", name="block1_conv1"),
                Conv2D(32,5,1,padding="same",
                    activation="relu", name="block1_conv2"),
                MaxPool2D(2, name="block1_pool"),
                Conv2D(128,5,1,padding="same",
                   activation="relu", name="block2_conv1"),
                Conv2D(128,5,1,padding="same",
                   activation="relu", name="block2_conv2"),
                MaxPool2D(2, name="block2_pool"),
                Flatten(),
                Dense(self.hidden_dim, activation="relu"),
                Dropout(0.3),
                Dense(self.num_classes, activation="softmax")
       ]


    def call(self, inputs):
        """
        Runs a forward pass on an input batch of images.
        
        :param inputs: 
        :return: logits 
        """
        l = inputs
        for layer in self.architecture:
            l = layer(l)
            
        return l


    def loss(self, logits, labels): 
        """
        Calculates the model cross-entropy loss after one forward pass.
        Softmax is applied in this function.
        
        :param logits: during training, a matrix containing the result of multiple convolution and feed forward layers
        :param labels: during training, matrix containing the train labels
        :return: the loss of the model as a Tensor
        """

        return tf.reduce_mean(tf.keras.losses.categorical_crossentropy(labels, logits, from_logits=True))

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels 
        
        :param logits: 
        :param labels: 
        
        :return: the accuracy of the model as a Tensor
        """

        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

