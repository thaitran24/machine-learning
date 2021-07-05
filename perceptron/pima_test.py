import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt

# inputs, targets, weights, activations, learning_rate
# activations[N x n] = inputs[N x m] * weights [m x n]
# weights[m x n] = weights[m x n] - learning_rate * (targets [N x n] - activations[N x n]) * inputs.T [m x N]
# training & recall

pima = np.loadtxt('E:/ML/pima-indians-diabetes.csv', delimiter=',')
xdim = pima.shape[0]    # number of rows
ydim = pima.shape[1]    # number of columns

print(pima.shape)

"""cleaning data"""
# person that pregnant more than 8 times recognise as 8
pima[np.where(pima[:, 0] > 8), 0] = 8
# group age
# 20-29: 1
# 30-39: 2
# 40-49: 3
# >= 50: 4
pima[np.where((pima[:, 7] >= 20) & (pima[:, 7] < 30)), 7] = 1
pima[np.where((pima[:, 7] >= 30) & (pima[:, 7] < 40)), 7] = 2
pima[np.where((pima[:, 7] >= 40) & (pima[:, 7] < 50)), 7] = 3
pima[np.where(pima[:, 7] >= 50), 7] = 4

targets = pima[:, ydim - 1:ydim]    # separate targets from data

# normalisation
pima = (pima - pima.mean(axis=0)) / pima.var(axis=0)

precision_rates = [0] * 8

for col in range(8):
    # separate inputs from data
    inputs = np.concatenate((pima[:xdim, :ydim - col - 2], pima[:xdim, ydim - col - 1:ydim - 1]), axis=1)
    pcn = Perceptron(inputs, targets)
    precision_rates[col] = pcn.train(100, 0.05)

plt.plot(precision_rates)
plt.title('max precision rates when removing each column')
plt.show()

inputs = pima[:xdim, :ydim - 1]
pcn = Perceptron(inputs, targets)
for i in range(8):
    precision_rates[i] = pcn.train(100, 0.05)

plt.plot(precision_rates, color='green', linestyle='dashed', linewidth = 3,
         marker='o', markerfacecolor='blue', markersize=12)
plt.title('max precision rates')
plt.show()



