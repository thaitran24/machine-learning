# Code for Perceptron model
# Learning from Machine Learning: An Algorithmic Perspective book
# Created by Thai Tran

import numpy as np


class Perceptron:
    """Simple perceptron"""
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        self.weight = np.random.random((self.inputs.shape[1] + 1, 1))

    def train(self, n_iterations, learning_rate):
        # create activations output
        activations = np.array((self.inputs.shape[0], self.weight.shape[1]))

        # insert bias -1 to input
        inputs = np.concatenate((-np.ones((self.inputs.shape[0], 1)), self.inputs), axis=1)

        plt_rates = [0] * n_iterations  # all rates in every loop

        for i in range(n_iterations):
            # calculate following activation function
            activations = np.dot(inputs, self.weight)

            # determine if output is 0 or 1 by comparing to 0
            activations = self.predict(activations)

            # calculate new weights
            self.weight -= learning_rate * np.dot(inputs.T, activations - self.targets)

            precision_rate = 0  # precision for each loop

            # check each output
            for j in range(len(activations)):
                for k in range(len(activations[0])):
                    if activations[j][k] == self.targets[j][k]:
                        precision_rate += 1

            # calculate precision for each loop
            precision_rate = precision_rate / (activations.shape[0] * activations.shape[1])
            plt_rates[i] = precision_rate

        print('max precision rate = ', max(plt_rates))
        # plt.hist(plt_rates)
        # plt.title('Precision rate')
        # plt.show()
        return max(plt_rates)

    def predict(self, activations):
        return np.where(activations > 0, 1, 0)



