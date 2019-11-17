# NeuralNetworks_DeepLearning
Repository that will contain my projects for the course of Neural Networks and Deep Learning attended in 2019 at University of Padua.

## Ex1: Implementing a Neural Network in numpy
In this work I present my implementation of a Neural Network (NN) with two layers and a framework (grid search) to search systematically for the best architecture and hyper-parameters (HPs) between some possible options defined with some a priori knowledge and heuristics. Finally I present the best model found and its performance on a simple dataset.

The pourpose of the project is to understand the complexity of building a deep learning framework, so that once one will start to use frameworks like TensorFlow or PyTorch will have more insights on their functioning.

## Ex2: Predicting MNIST characters with Feed-Forward Neural Network
In this work I present my implementation of a Neural Network (NN) in Pytorch and a framework to tune the hyper-parameters (HPs) of the network with a random search based on some prior probability distributions. 
I then show how to use two consecutive random searches to find the best model and discuss its performance. Finally I present some techniques that can help in visualizing what each neuron is learning.

The notable aspects of this project are:
* The use of a random search with prior distributions;
* The introduction of a method to create a prior distribution for sampling the number of hidden layers and the number of hidden neurons given the input and output dimension and the compression rate (e.g. the ratio between neurons in one layer and the previous one) that we want to hold in general;
* The introduction of a technique that I call activation map to visualize which parts of an input are responsible for the excitation or the inhibition of each neuron.
