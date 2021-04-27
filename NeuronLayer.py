#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

np.random.seed(42)

class NeuronLayer:
    
    """This creates a layer for the network hidden as well as output layer """
    def __init__(self, n_input, n_neurons):
    
        self.weights = np.random.rand(n_input, n_neurons)
        self.bias = np.random.rand(n_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        
# activation  function for classification
    def activate(self, x):
        r = np.dot(x, self.weights) + self.bias
        self.last_activation = self.sigmoid(r)
        return self.last_activation
    
# sigmoid as a activation function for feed forward
    def sigmoid(self, r):
        return 1 / (1 + np.exp(-r))

#derivative of sigmoid for backward propagation
    def sigmoid_derivative(self, r):
        return r * (1 - r)

