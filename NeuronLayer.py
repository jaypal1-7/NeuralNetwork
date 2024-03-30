import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

np.random.seed(42)

class NeuronLayer:
    """Creates a layer for the network, serving as both hidden and output layer."""
    def __init__(self, input_size, num_neurons):
        self.weights = np.random.rand(input_size, num_neurons)
        self.bias = np.random.rand(num_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        
    # Activation function for classification
    def activate(self, inputs):
        result = np.dot(inputs, self.weights) + self.bias
        self.last_activation = self.sigmoid(result)
        return self.last_activation
    
    # Sigmoid as the activation function for feed forward
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Derivative of sigmoid for backward propagation
    def sigmoid_derivative(self, x):
        return x * (1 - x)

