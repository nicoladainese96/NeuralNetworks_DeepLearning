# NeuralNetworks_DeepLearning
Repository that will contain my projects for the course of Neural Networks and Deep Learning attended in 2019 at University of Padua.

## Ex1: Implementing a Neural Network in numpy
In this work I present my implementation of a Neural Network (NN) with two layers and a framework (grid search) to search systematically for the best architecture and hyper-parameters (HPs) between some possible options defined with some a priori knowledge and heuristics. Finally I present the best model found and its performance on a simple dataset.

The pourpose of the project is to understand the complexity of building a deep learning framework, so that once one will start to use frameworks like TensorFlow or PyTorch will have more insights on their functioning.

<img src=Supplementary_material/learned_function.png>

## Ex2: Predicting MNIST characters with Feed-Forward Neural Network
In this work I present my implementation of a Neural Network (NN) in Pytorch and a framework to tune the hyper-parameters (HPs) of the network with a random search based on some prior probability distributions. 
I then show how to use two consecutive random searches to find the best model and discuss its performance. Finally I present some techniques that can help in visualizing what each neuron is learning.

The notable aspects of this project are:
* The use of a random search with prior distributions;
* The introduction of a method to create a prior distribution for sampling the number of hidden layers and the number of hidden neurons given the input and output dimension and the compression rate (e.g. the ratio between neurons in one layer and the previous one) that we want to hold in general;
* The introduction of a technique that I call activation map to visualize which parts of an input are responsible for the excitation or the inhibition of each neuron.

<img src=Supplementary_material/activationPF.png>

## Ex3: Text generation with LSTM
In this project I implemented in Pytorch a Recurrent Neural Network with Long Short Term Memory (LSTM) cells working at word-
level. I then trained the network on some books of Jane Austen and finally generated some sample text providing different sequences as context seed.

The results are not super exciting, but not that bad either. As an example I generated some text from the famous incipit of the novel Pride and Prejudice "It is a truth universally acknowledged":

"by us for you have been thorough in getting at the park . 
connexions or caution , to please before in dinner for him in his very voice with the time and visitors was taken up striking
, french , till lady their william was well claim . 
in making her aunt , till this had her interest , because mr allen remained simile they had mr crawford , he spoke her from the 
girls said exercise to expect in her way , i shall set out of her memory , every awkward reproved in that countenance , mr
dashwood made to comprehend from the subject , where renewed such years and beauty , how can i know the winter and from the same light . 
her duty was too far the smallest opportunity , and as a complacency to be made to do i have been all the touch of penetration with saying . . 
meanwhile but you thought what you and yourself they are shocked . 
, to forgive , nothing cannot be indeed taking our notice of our power . "

I would say that there is a good understanding of the grammatical structure of a phrase, but almost no understanding of the semantic meaning of the words, so overall the generated text doesn't make any sense. I also noticed that sometimes a comma is placed after a point. In hindsight it is just my fault, because I splitted the text in entire sequences, but never trained on sequences that spanned across multiple phrases, so the point was always at the end and that makes impossible to predict the next word.

## Ex4: Compressing MNIST dataset with autoencoders
In this project I implemented in Pytorch an Autoencoder, tuning its parameters through a random search and training it on the 
MNIST dataset. I then explored the compression-performance trade-off, varying the dimension of the encoded space of the 
autoencoder, and testing for different dimensions the performance on the test set in three different setups: using the test set
uncorrupted, adding to it gaussian noise or occluding part of the images. I then repeated the same testing procedure on 
denoising autoencoders. Finally I implemented a method for generating images with the Autoencoder sampling from the encoded 
space and analysed the smoothness and homogeneity properties of a bi-dimensional encoded space.

I invite you to play around in the notebook Nicola_Dainese_Ex4_NN_pt3_EncodedSpace.ipynb in the folder Nicola_Dainese_Ex4_NN to see an interactive 3D plot of the encoded space smoothness and homogeneity.

To test the smoothness and the homogeneity of the encoded space I defined a function D(x,y) that is equal to the square distance in the output space from the image of the nearest centroid in the encoded space.

For example if I take the centroid of the digit 1, the nearest centroid is the same as the point itself, so its function D will be 0. Now if we take the nearest centroid different from 1 and we move toward it, the function D will increase because we are changing the optput and the distance is non-negative. When the point selected is halfway between the two nearest centroids we naturally reach a maximum, but it can be continuous or discontinuous, depending on whether the distance in the output space from the two centroids is the same or not.

Looking from above the plot it's possible to see the "basins of attraction" of all the different digits, delimited by the ridges. The orange dots represent the centroids of the 10 digits and hovering over them is possible to see which digit corresponds to each of them. We can see that 4 and 9 are mapped adjacent to one another and so are 3, 5 and 8. Also it's possible to see that the 2D encoded space of an autoencoder is smooth but not homogeneous, since ridges are not smooth.

<img src=Supplementary_material/AE_latent.png>

## Ex5: Reinforcement learning with tabular methods
In this project I tested two different reinforcement learning (RL) algorithms from the tabular methods class: SARSA and Q-learning.
Actually this was very basic assignment that required less work than the previous ones and I already had mature project on RL in my portfolio, so if you're interested I suggest you check it out at https://github.com/nicoladainese96/haliteRL .