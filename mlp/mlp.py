
import numpy as np


class MLP:
    """ MLP with 1 hidden layer """

    def __init__(self, inputs, targets, n_hidden, learning_rate=0.25,
                 beta=1, momentum=0.9, out_type='logistic'):
        # set network properties
        self.eta = learning_rate
        self.beta = beta
        self.momentum = momentum
        self.out_type = out_type

        # set network size
        self.n_hidden_nodes_weights = n_hidden
        self.n_in = inputs.shape[1]
        self.n_out = targets.shape[1]
        self.n_data = inputs.shape[0]

        self.weights_hidden = np.random.rand(self.n_in + 1, self.n_hidden_nodes_weights)
        self.weights_output = np.random.rand(self.n_hidden_nodes_weights + 1, self.n_out)

    def train(self, inputs, targets, n_iterations):
        new_inputs = np.concatenate((inputs, np.ones((inputs.shape[0], 1))), axis=1)
        update_output = np.zeros((np.shape(self.weights_output)))
        update_hidden = np.zeros((np.shape(self.weights_hidden)))

        for i in range(n_iterations):
            print('iteration ', i)
            output = self.forward(new_inputs)

            if self.out_type == 'logistic':
                delta_output = self.beta * (output - targets) * output * (1.0 - output)
            if self.out_type == 'linear':
                delta_output = (output - targets) / self.n_data
            if self.out_type == 'softmax':
                delta_output = (output - targets) * (output * (-output) + output) / self.n_data

            delta_hidden = self.beta * self.hidden * (1.0 - self.hidden) * np.dot(delta_output, self.weights_output.T)

            update_output = self.eta * np.dot(self.hidden.T, delta_output) + self.momentum * update_output
            update_hidden = self.eta * np.dot(new_inputs.T, delta_hidden[:, :-1]) + self.momentum * update_hidden
            self.weights_output -= update_output
            self.weights_hidden -= update_hidden

            if self.out_type == 'logistic':
                res = np.where(output > 0.5, 1, 0)
                error = np.sum((res - targets) ** 2)
                print('error = ', error / (np.sum(targets ** 2)))
                if error == 0:
                    break

    def forward(self, inputs):
        self.hidden = np.dot(inputs, self.weights_hidden)
        self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))
        self.hidden = np.concatenate((self.hidden, np.ones((inputs.shape[0], 1))), axis=1)

        output = np.dot(self.hidden, self.weights_output)

        if self.out_type == 'logistic':
            return 1.0 / (1.0 + np.exp(-self.beta * output))
        if self.out_type == 'linear':
            return output
        if self.out_type == 'softmax':
            normaliser = np.sum(np.exp(output), axis=1) * np.ones((1, output.shape[0]))
            return np.transpose(np.transpose(np.exp(output)) / normaliser)
        else:
            print('error')
