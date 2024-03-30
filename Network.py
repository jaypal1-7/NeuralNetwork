#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

class Network:
    """Creates a network by connecting layers and adjusting weights and biases."""
    
    def __init__(self):
        self._layers = []
        
    def add_layer(self, layer):
        """Adds a layer to the network."""
        self._layers.append(layer)
        
    def feed_forward(self, X):
        """Performs feed forward using input data, weights, and activation functions."""
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):
        """Predicts the output based on the input data."""
        return self.feed_forward(X)
      
    def backward_propagation(self, X, y):
        """Performs backward propagation to update weights based on error."""
        learning_rate = 0.1
        output = self.feed_forward(X)
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]
            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.sigmoid_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.sigmoid_derivative(layer.last_activation)
        for i in range(len(self._layers)):
            layer = self._layers[i]
            input_use = np.atleast_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_use.T * learning_rate
            
    def train(self, X, y, n_epochs=200):
        """Trains the neural network."""
        mses = []
        for i in range(n_epochs):
            for j in range(len(X)):
                self.backward_propagation(X[j], y[j])
                mse = np.mean(np.square(y - self.feed_forward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))
        return mses
    
    @staticmethod
    def accuracy(y_pred, y_true):
        """Calculates the accuracy between predicted labels and true labels."""
        return ((np.round(y_pred, 1) == y_true)).mean()

class NeuronLayer:
    """Creates a layer for the network, serving as both hidden and output layer."""
    
    def __init__(self, input_size, num_neurons):
        self.weights = np.random.rand(input_size, num_neurons)
        self.bias = np.random.rand(num_neurons)
        self.last_activation = None
        self.error = None
        self.delta = None
        
    def activate(self, inputs):
        """Applies activation function to the inputs."""
        result = np.dot(inputs, self.weights) + self.bias
        self.last_activation = self.sigmoid(result)
        return self.last_activation
    
    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid function."""
        return x * (1 - x)

if __name__ == '__main__':
    nn = Network()
    nn.add_layer(NeuronLayer(7, 5))
    nn.add_layer(NeuronLayer(5, 1))
    
    # Read and preprocess the data
    filepath = "your_dataset_filepath_here"
    a = pd.read_csv(filepath, sep='\n', lineterminator='\r')
    a = list(a)
    b = pd.DataFrame([z.split('\t') for z in a])
    x = b.iloc[:, :7].dropna()
    X = np.array(x)
    Y = b.iloc[:, 7:8]
    y = np.array(Y)
    
    # Train the neural network
    errors = nn.train(X, y, 300)
    
    # Display results
    print("Accuracy: %.2f%%" % (nn.accuracy(nn.predict(X)[:, 0].T, y.flatten()) * 100))
    print("Data output:\n" + str(y))
    print("Predicted output:\n" + str(nn.predict(X)))
    print("Predicted output rounded:\n" + str(np.round(nn.predict(X), 1)))
    
    # Plot changes in MSE
    plt.plot(errors, c='b', label='MSE')
    plt.title('Changes in MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()
