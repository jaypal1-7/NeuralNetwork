# NeuralNetwork
It is an implementation of neural network in python usnig sigmoid activation function.

# Overview
This a simple neural network with single hidden layer consisting of 5 neurons. To train the model, seed_dataset and iris_dataset are used. Weight and bias randomly created using numpy. sigmoid function is used as activation function for feed forward. Sigmoid derivative has been used for backward propagation. The  model has been trained several times to adjust its weight and bias inorder to provide better classification results. Here the output is class of seed and irirs plant.

# Dataset
Two datasets have been used to train the model for classification. This is the link to UCI machine learning repository 
https://archive.ics.uci.edu/ml/datasets/iris
https://archive.ics.uci.edu/ml/datasets/seeds

# Functioning
1) Class NeuronLayer is defined for creation of layer parameters like weight, no. of neurons in layer and bias.
2) Sigmoid function is defined as the activation function in NeuronLayer class, which takes matrix multiplication of weights and input from previous layer.
3) Class Network is defined where the network is buid using NeuronLayer class with the help of add function.
4) feed_forward functions gives output according to the input and weights for the classification.
5) backward_propagation changes the weight and bias according to the error in predicted and actual value of target.
6) One cycle of feed forward and backwardpropagation is called one epoch and thye network is trained 300 times to fine tune the model for better classification.
7)The final results are close to the actual ones after the training of model.


# stucture
Class NeuronLayer(NeuronLayer.py) : This class form a sinlge layer of neurons. It is used by Class Network for bulding the neural network.
Class Network(Network.py) : This class connects layers,trains model on the data using performs feed_forward and backward_propagation. 
Main funtion: In the Main function, data is fetched from csv file and after formating used for training the model.






