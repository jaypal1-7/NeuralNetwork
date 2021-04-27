#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO

# Layer formation from Input data and connecting layers
class Network:
    
    """ Layers are connected and weight and bias have been adjusted using funtions of this class"""
    
    def __init__(self):
        self._layers = []
#adding layers and append it in self._layer
    def add_layer(self, layer):
        self._layers.append(layer)
        
#feed forward using input, weights and activation function
    def feed_forward(self, X):
        for layer in self._layers:
            X = layer.activate(X)
        return X

    def predict(self, X):

        ff = self.feed_forward(X)
        return ff
      
    def backward_propagation(self, X, y):
        learning_rate = 0.1
       #output from  feed forward
        output = self.feed_forward(X)

        # Loop over the layers backward
        # to calculate the error to update weights
        for i in reversed(range(len(self._layers))):
            layer = self._layers[i]

            if layer == self._layers[-1]:
                layer.error = y - output
                layer.delta = layer.error * layer.sigmoid_derivative(output)
            else:
                next_layer = self._layers[i + 1]
                layer.error = np.dot(next_layer.weights, next_layer.delta)
                layer.delta = layer.error * layer.sigmoid_derivative(layer.last_activation)
                
        #update weights according to the error in ouput
        for i in range(len(self._layers)):
            layer = self._layers[i]
            input_use = np.a_2d(X if i == 0 else self._layers[i - 1].last_activation)
            layer.weights += layer.delta * input_use.T * learning_rate
            
    def train(self, X, y, n_epochs=200):
        mses = []
        for i in range(n_epochs):
            for j in range(len(X)):
                self.backward_propagation(X[j], y[j])
                mse = np.mean(np.square(y - nn.feed_forward(X)))
                mses.append(mse)
                print('Epoch: #%s, MSE: %f' % (i, float(mse)))

        return mses
    
# Calculates the accuracy between the predicted labels and true labels.
    @staticmethod
    def accuracy(y_pred, y_true):
        return ((np.round(y_pred,1)== y_true)).mean()

            
# no. of features in dataset is 7 so input is 7 and no. of neurons is 5 for hidden layer
#and output layer have 1 neuron and 5 input coming from hidden layer.
if __name__ == '__main__':
    nn = Network()
    nn.add_layer(NeuronLayer(7, 5))
    nn.add_layer(NeuronLayer(5, 1))
    
#Here thye network is trained on seed_dataset which have 7 features for corn classsifiction
#It is also experimentesd with iris data set
    a = pd.read_csv(filepath,sep = '\n', lineterminator='\r')
    a = list(a)

    b = pd.DataFrame([z.split('\t') for z in a])
    x = b.iloc[:,:7]
    x = x.dropna()
    X = np.array(x)
    Y = b.iloc[:,7:8]
    y = np.array(Y)
    
#Train the neural network and predict the data
    errors = nn.train(X, y, 300)
    print("Accuracy: %.2f%%" % (nn.accuracy(nn.predict(X)[:,0].T, y.flatten()) * 100))
    print("Data ouput: \n" + str(y))
    print("Predicted output: \n" + str(nn.predict(X)))
    print("Predicted output round: \n" + str(np.round(nn.predict(X),1)))

    # Plot changes in mse
    plt.plot(errors, c = 'b', label = 'MSE')
    plt.title('Changes in MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.grid(linestyle='-.', linewidth=0.5)
    plt.legend()
    plt.show()

