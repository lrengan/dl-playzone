# code from Michael Nielsen's book http://neuralnetworksanddeeplearning.com/index.html
import random

import numpy as np
#from __builtin__ import xrange


class Network(object):

    # ex. net = Network([2, 3, 1])
    # defines 3 layers with 2 neurons in 1st layer, 3 in 2nd and 1 in the output layer
    def __init__(self, sizes):

        self.num_layers = len(sizes)

        self.sizes = sizes

        # self.biases is a list of numpy matrices, where each matrix contains
        # normal distributed (mean=0, std=1) random values for biases
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]

        # self.weights is a list of numpy matrices, where each matrix contains
        # normal distributed (mean=0, std=1) random values for weights
        # Let W = self.weights; W[0] holds the weights between layer 2 and layer 3
        # W[1][j,k] is the weight between k -> j, k-th neuron in 2nd layer to
        # j-th neuron in 3rd layer
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    # end __init__

    # do one feed forward computation
    def feed_forward(self, a):
        #  compute the output of the network for a given input a
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        return a
    # end feed_forward

    # Stochastic Gradient Descent
    def SGD(self, train_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch stochastic
            gradient descent.  The "training_data" is a list of tuples
            "(x, y)" representing the training inputs and the desired
            outputs.  The other non-optional parameters are
            self-explanatory.  If "test_data" is provided then the
            network will be evaluated against the test data after each
            epoch, and partial progress printed out.  This is useful for
            tracking progress, but slows things down substantially."""

        if test_data:
            n_test = len(test_data)
        n = len(train_data)
        for j in range(epochs):
            # shuffle the input data
            random.shuffle(train_data)
            # construct and iterate over mini-batches
            mini_batches = [train_data[k:k+mini_batch_size]
                            for k in range(0,n,mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data :
                print "Epoch {0} : {1} / {2}".format(j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} : complete".format(j)

    # end SGD()

# end class Network


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
