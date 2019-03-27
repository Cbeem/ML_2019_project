# NNlib, the Neural Network Library of project group 76
# This library provides numerous functionalities to build your own Neural Network!


import random
import math
import numpy as np
import pickle  # So we can save the model
from PrintColors import *
import matplotlib.pyplot as plt


def sigmoid(layer):
    """ Performs the sigmoid activation function on the given layer."""
    return 1 / (1 + np.exp(-layer))


def sigmoid_der(activation_value):
    """ Computes the (sigmoid) derivative of the given layer."""
    return np.multiply(activation_value, 1 - activation_value)


def relu(activation_value):
    """ Performs the ReLu activation function of the given layer."""
    return np.maximum(activation_value, 0, activation_value)


def relu_der(activation_value):
    """ Computes the (relu) derivative of the given activation_value(s)."""
    der = np.copy(activation_value)

    " Code source: https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy/46411340"
    der[der <= 0] = 0
    der[der > 0] = 1
    return der


def tanh(layer):
    """ Performs the tanh activation function on the given layer."""
    return 2 / (1 + np.exp(-2 * layer)) - 1


def tanh_der(layer):
    """ Computes the (tanh) derivative of the given activation_value(s)."""
    return 1 - np.power(layer, 2)


def softmax(layer):
    """ Applies the softmax funciton on an numpy array/matrix."""
    exponent = np.exp(layer)
    return exponent / np.sum(exponent)


def cross_entropy(predictions, targets, epsilon=1e-9, verbose=False):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    pos = targets * np.log(predictions + 1e-9)
    neg = np.subtract(1, targets) * (np.log(np.subtract(1, predictions)) + 1e-9)

    ce = -np.sum((pos + neg))
    if verbose:
        print("|CROSS ENTROPY| VERBOSE:", ce, predictions, targets, pos, neg)
    return ce


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot(data, ma=1, x_lab="#samples", y_lab="ce error"):
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    print(data)
    for values, label in data:
        print(values)
        print(label)
        ax.plot(moving_average(values, ma), label=label)

    plt.legend()
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.show()
    plt.axis('scaled')


class NeuralNet:
    """ This class allows you to create a neural network.
        Note that the layers are created in the order in which they are provided.
        Thus you must first provide the input layer, then (optionally) the hidden layer(s) and then the output layer.
        Once the output layer is defined, no new layers can be added."""

    def __init__(self):
        self._num_of_features = -1

        # An array containing zero or more numpy arrays (layers).
        # Note that these layers are generated through the forward pass, not at there definition.
        self._layers = []
        self._num_of_outputs = -1

        # Two layers are connected with a matrix of weights. For three or more layers, we need two or more
        # weight matrices. All these weight matrices are stored in the 'weights' array below.
        self._weights = []  # An array containing one or more numpy matrices.

        self._learning_rates = []  # Each layer has its own learning rate.
        self._activation_functions = []  # The activation function is configurable per hidden layer.

        self._ACT_SIGMOID = 0  # We use these variables to check which activation function we should apply.
        self._ACT_RELU = 1  # These variables are like class constants.
        self._ACT_TANH = 2

    def add_input_layer(self, num_of_features):
        """ Defines the input layer with num_of_features columns.

            Note: The data for this layer must be provided at each forward pass. This function defines the size of
                  the matrix so that the weight matrices can be defined."""

        if self._num_of_features > 0:
            print(bcolors.FAIL + "Error: The input layer was already defined." + bcolors.ENDC)
            exit(1)

        # We add a non-used 'activation function' and 'learning rate' to the array to align the index of this array with
        # the index of the hidden layers.
        self._activation_functions.append(-1)
        self._learning_rates.append(-1)

        # Simply specify the dimensions so that the weight matrices can be created.
        self._num_of_features = num_of_features

        # Add an 'empty' layer as the input layer
        empty_layer = np.matrix([0 for _ in range(num_of_features + 1)])
        self._layers.append(empty_layer)

    def add_hidden_layer(self, num_of_nodes, activation_function="Sigmoid", learning_rate=0.001):
        """ Adds a hidden layer with num_of_nodes nodes to the network. Weights are automatically generated and added
            to the network to connect this layer to the previous one.

            Note: - The input layer must be defined before this function is called.
                  - It is not possible to use this function after the output layer is defined.
                  - The activation function and learning rate are optional and per layer configurable."""

        if self._num_of_features < 0:
            print(bcolors.FAIL + "Error: The input layer is not yet defined." + bcolors.ENDC)
            exit(1)

        if self._num_of_outputs > 0:
            print(bcolors.FAIL + "Error: The output layer was already defined." + bcolors.ENDC)
            exit(1)

        if activation_function == "Sigmoid" or activation_function == "sigmoid":
            self._activation_functions.append(self._ACT_SIGMOID)
        elif activation_function == "Relu" or activation_function == "relu" or activation_function == "ReLu":
            self._activation_functions.append(self._ACT_RELU)
        elif activation_function == "Tanh" or activation_function == "tanh":
            self._activation_functions.append(self._ACT_TANH)
        else:
            print(bcolors.FAIL + "Error: activation function '{}' is not defined.".format(activation_function) +
                  bcolors.ENDC)
            exit(1)

        self._learning_rates.append(learning_rate)

        # The to be added weight matrix is of size N x M, where N is the number of column from the previous layer
        # And M num_of_nodes. Thus we need to extract the number of columns from the previous layer, which
        # will be the number of rows for the to be added weight matrix
        num_of_rows = np.size(self._layers[-1], 1)

        # The weights are sampled from a Glorot uniform distribution with the range defined below.
        weights_range = math.sqrt(6 / (num_of_rows + num_of_nodes))

        # This can be done on a single line, but that looks nasty.
        weight_matrix = []
        for _ in range(num_of_rows):
            # Appends num_of_features rows to weight matrix
            weight_matrix.append([random.uniform(-weights_range, weights_range)
                                  for _ in range(num_of_nodes)])
        self._weights.append(np.matrix(weight_matrix))

        empty_layer = [0 for _ in range(num_of_nodes + 1)]  # +1 for the bias node
        self._layers.append(np.matrix(empty_layer))  # Adding a None allows us to index the array at the forward pass

    def add_output_layer(self, num_of_outputs, learning_rate=0.01):
        """ Defines the output layer of the neural network. Weights are automatically generated and added to the
            network to connect the output layer to the previous layer.
            Note: - It is not necessary to have hidden layers to create and output layer.
                  - It IS necessary to have an input layer defined.
                  - The learning rate is optional and configurable."""

        if self._num_of_features < 0:
            print(bcolors.FAIL + "Error: The input layer is not yet defined." + bcolors.ENDC)
            exit(1)

        self._learning_rates.append(learning_rate)

        self._num_of_outputs = num_of_outputs

        # The to be added weight matrix is of size N x M, where N is the number of column from the previous layer
        # And M num_of_nodes. Thus we need to extract the number of columns from the previous layer, which
        # will be the number of rows for the to be added weight matrix
        num_of_rows = np.size(self._layers[-1], 1)

        # The weights are sampled from a Glorot uniform distribution with the range defined below.
        weights_range = math.sqrt(6 / (num_of_rows + num_of_outputs))

        # This can be done on a single line, but that looks nasty
        weight_matrix = []
        for _ in range(num_of_rows):
            # Appends num_of_features rows to weight matrix
            weight_matrix.append([random.uniform(-weights_range, weights_range)
                                  for _ in range(num_of_outputs)])

        self._weights.append(np.matrix(weight_matrix))

        empty_layer = [0 for _ in range(num_of_outputs)]
        self._layers.append(np.matrix(empty_layer))  # Adding a None allows us to index the array at the forward pass

    def forward_pass(self, features):
        """ Performs a forward pass through the network, given the features.
            You must provide the features of the entire input layer, and the number of features must match the number
            of nodes of the input layer (excluding the bias node)."""

        if self._num_of_features < 0:
            print(bcolors.FAIL + "Error: Input layer must be defined before doing a forward pass" + bcolors.ENDC)
            exit(1)

        if self._num_of_outputs < 0:
            print(bcolors.FAIL + "Error: Output layer must be defined before performing a forward pass" + bcolors.ENDC)
            exit(1)

        activation_value = 0
        self._layers[0] = np.matrix(np.append(features, [1]))  # [1] adds the bias node

        # Loop through all remaining layers, excluding the output layer
        for layer_index in range(1, len(self._layers) - 1):
            # Check which activation function to apply, then:
            # The activation values of the current layer are the activation values of the previous layer
            # times the weight matrix and passed through the activation function.
            if self._activation_functions[layer_index] == self._ACT_SIGMOID:
                activation_value = sigmoid(np.dot(self._layers[layer_index - 1], self._weights[layer_index - 1]))
            elif self._activation_functions[layer_index] == self._ACT_RELU:
                activation_value = relu(np.dot(self._layers[layer_index - 1], self._weights[layer_index - 1]))
            elif self._activation_functions[layer_index] == self._ACT_TANH:
                activation_value = tanh(np.dot(self._layers[layer_index - 1], self._weights[layer_index - 1]))
            else:
                print(bcolors.FAIL + "Error: tried to use a undefined activation function in the forward pass" +
                      bcolors.ENDC)
                exit(1)
            self._layers[layer_index] = np.append(activation_value, [[1]], axis=1)  # Append the bias node

        # Go to the final layer and determine its logits (its un-normalized inputs).
        self._layers[-1] = np.dot(self._layers[-2], self._weights[-1])

        # Normalize the output by applying the softmax function.
        self._layers[-1] = softmax(self._layers[-1])

        return np.argmax(self._layers[-1], axis=1)

    def forward_pass_loss(self, features, target):
        result = self.forward_pass(features)
        loss = cross_entropy(np.array(self._layers[-1]), np.array(target))
        return result, loss


    def backward_pass(self, target):
        """ Backpropogate through the neural network to update the weights.
            You must provide the target values of the entire output layer.
            - The derivatives of the activation functions defined for the hidden layers are used.
            - The learning rates depend on which is defined (or the default) at the creation of the network."""
        global_loss = prev_loss = np.matrix(self._layers[-1] - target)

        # Go through the network in reversed order.
        for index in reversed(range(1, len(self._layers) - 1)):
            prev_loss_t = prev_loss.T           # Transpose 'next' layer output derivatives
            current_layer_loss_out = np.dot(self._weights[index], prev_loss_t).T  # Current layer loss derivatives

            act_value_t = self._layers[index].T  # Current layer activation values
            # The weight gradient for the weights between this layer and the layer ABOVE
            # Learning rate + 1 because it's the layer ABOVE that wants to update its weights.
            weights_update = np.dot(act_value_t, prev_loss) * self._learning_rates[index + 1]

            self._weights[index] = np.subtract(self._weights[index], weights_update)  # Update next layer weights
            if self._activation_functions[index] == self._ACT_SIGMOID:  # Check which activation function was applied.
                prev_loss = np.multiply(current_layer_loss_out, sigmoid_der(self._layers[index]))
            elif self._activation_functions[index] == self._ACT_RELU:
                prev_loss = np.multiply(current_layer_loss_out, relu_der(self._layers[index]))
            elif self._activation_functions[index] == self._ACT_TANH:
                prev_loss = np.multiply(current_layer_loss_out, tanh_der(self._layers[index]))
            else:
                print(bcolors.FAIL + "Error: tried to use a undefined activation function at the backward_pass" +
                      bcolors.ENDC)
                exit(1)
            prev_loss = np.delete(prev_loss, -1)

        input_t = self._layers[0].T
        weights_update = np.dot(input_t, prev_loss) * self._learning_rates[1]
        self._weights[0] = np.subtract(self._weights[0], weights_update)
        return global_loss

    def print_network(self, verbose=True):
        """ Prints the entire network."""

        print("\n")
        print(bcolors.UNDERLINE + bcolors.BOLD + "Note that the bias node adds one node to each non-output layer" +
              bcolors.ENDC)

        index = 0
        if not verbose:  # Prints a summary
            for index, weights in enumerate(self._weights):
                print("Layer {}: \t\t\t".format(index), np.shape(self._layers[index]), " rows x columns")
                print("Weights {} -> {}: \t".format(index, index + 1), np.shape(weights), " rows x columns")

            print("Output Layer: \t\t".format(index + 1), np.shape(self._layers[index + 1]), " rows x columns")
            return

        print("")
        for index, weights in enumerate(self._weights):  # Also prints the content of each layer
            print(bcolors.HEADER +
                  "---------------------------------------------------------\n"
                  "                         Layer {}\n"
                  "---------------------------------------------------------".format(index) + bcolors.ENDC)
            print("Shape: ", np.shape(self._layers[index]), " rows x columns")
            print(self._layers[index])

            print(bcolors.HEADER + "\n\n"
                  "---------------------------------------------------------\n"
                  "             Weights connecting layer {} to {}\n"
                  "---------------------------------------------------------".format(index, index + 1) + bcolors.ENDC)
            print("Shape: ", np.shape(weights), " rows x columns")
            print(weights)
            print("\n\n")

        print(bcolors.HEADER +
              "---------------------------------------------------------\n"
              "                    layer {} (Output)\n"
              "---------------------------------------------------------".format(index + 1) + bcolors.ENDC)
        print("Shape: ", np.shape(self._layers[index + 1]), " rows x columns")
        print(self._layers[index + 1])

    def print_layer_stats(self):
        """ Prints the configurations (#nodes, activation value, learning rate) used for each layer."""
        print("\n")
        print(bcolors.UNDERLINE + bcolors.BOLD + "Note that the bias node adds one node to each non-output layer" +
              bcolors.ENDC)

        index = 0
        for index, weights in enumerate(self._weights):
            activation_function = "Undefined"
            learning_rate = "Undefined" if self._learning_rates[index] == -1 else self._learning_rates[index]
            print("Layer {}: ".format(index), np.shape(self._layers[index]), " rows x columns")
            if self._activation_functions[index] == self._ACT_SIGMOID:
                activation_function = "Sigmoid"
            elif self._activation_functions[index] == self._ACT_RELU:
                activation_function = "ReLu"

            print("\t - Activation function: {}.".format(activation_function))
            print("\t - Learning rate: {}.".format(learning_rate))

        index += 1
        learning_rate = self._learning_rates[index]
        print("Output Layer: ".format(index), np.shape(self._layers[index]), " rows x columns")
        print("\t - Activation function: SoftMax.")
        print("\t - Learning rate: {}.".format(learning_rate))

    def save_network_to_disk(self, name, overwrite=False):
        """ Saves the current network to the disk, including all its configurations.
            This allows for a model to be saved."""

        if overwrite == True:
            with open('Models\\' + name + '.pkl', 'wb') as output:
                pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
        else:
            # 'Easier to ask for forgiveness than permission'. Source: https://docs.python.org/2/glossary.html#term-eafp
            # We let it crash on purpose to check if file exists.
            try:
                pickle.load(open('Models\\' + name + '.pkl', "rb"))
                print(bcolors.FAIL + "Error: the file '{}.pkl' already exists.".format(name) + bcolors.ENDC)
            except (OSError, IOError):
                with open('Models\\' + name + '.pkl', 'wb') as output:
                    pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_network_from_disk(name):
        """ Loads a network (model) from disk."""
        r = None
        try:
            r = pickle.load(open('Models\\' + name + '.pkl', "rb"))
        except (OSError, IOError):
            print(bcolors.FAIL + "The file '{}.pkl' does not exist.".format(name) + bcolors.ENDC)
            exit(1)
        return r
