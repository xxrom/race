import torch
import torch.nn as nn
import random

import numpy as np

CPUS = 6
torch.set_num_threads(CPUS * 4)
torch.set_num_interop_threads(CPUS)


# Neural_Network
class AI(nn.Module):

  def __init__(self, weights, layers=[2, 4, 4, 3, 3]):
    super(AI, self).__init__()

    self.weights = weights.copy()
    self.layers = layers.copy()

    # TODO: parameters can be parameterized instead of declaring them here
    self.inputSize = layers[0]
    self.hiddenSize0 = layers[1]
    self.hiddenSize1 = layers[2]
    self.hiddenSize2 = layers[3]
    # self.outputSize = layers[4]

    # weights
    self.setWeights(weights)

  def forward(self, X):
    # 3 X 3 ".dot" does not broadcast in PyTorch
    # MyX = torch.tensor(([X]), dtype=torch.float)

    # matrix multiplication
    self.z1 = X @ self.W0
    # activation function
    self.z2 = self.sigmoid(self.z1)

    self.z3 = self.z2 @ self.W1
    self.z4 = self.sigmoid(self.z3)

    self.z5 = self.z4 @ self.W2
    self.z6 = self.sigmoid(self.z5)

    return self.z6

    # self.z7 = torch.matmul(self.z6, self.W3)
    # final activation function
    # o = self.sigmoid(self.z7)

    # return o

  def sigmoid(self, s):
    return 1 / (1 + torch.exp(-s))

  def sigmoidPrime(self, s):
    # derivative of sigmoid
    return s * (1-s)

  def backward(self, X, y, o):
    self.o_error = y - o  # error in output
    self.o_delta = self.o_error * self.sigmoidPrime(
        o)  # derivative of sig to error
    self.z2_error = torch.matmul(self.o_delta, torch.t(self.W2))
    self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)
    self.W1 += torch.matmul(torch.t(X), self.z2_delta)
    self.W2 += torch.matmul(torch.t(self.z2), self.o_delta)

  def train(self, X, y):
    # forward + backward pass for training
    o = self.forward(X)
    self.backward(X, y, o)
    # self.xPredicted = o

  # def saveWeights(self, model):
  # we will use the PyTorch internal storage functions
  # torch.save(model, "NN")
  # you can reload model with all the weights and so forth with:
  # torch.load("NN")

  def setWeights(self, weights):
    self.weights = weights[:]

    # w0 = np.array(self.weights[0]).transpose().tolist()
    # w1 = np.array(self.weights[1]).transpose().tolist()
    # w2 = np.array(self.weights[2]).transpose().tolist()

    # self.W0 = torch.tensor(([w0]), dtype=torch.float)
    # self.W1 = torch.tensor(([w1]), dtype=torch.float)
    # self.W2 = torch.tensor(([w2]), dtype=torch.float)

    self.W0 = torch.tensor(([self.weights[0]]), dtype=torch.float)
    self.W1 = torch.tensor(([self.weights[1]]), dtype=torch.float)
    self.W2 = torch.tensor(([self.weights[2]]), dtype=torch.float)
    # self.W3 = torch.tensor(([self.weights[3]]), dtype=torch.float)

  def setWeightsByIndexes(self, value, i, j, k):
    self.weights[i][j][k] = value

    W = None

    if i == 0:
      W = self.W0

    if i == 1:
      W = self.W1

    if i == 2:
      W = self.W2

    if W is None:
      print('W is NONE!')
      return

    W[0][j][k] = value

  def getWeightByIndexes(self, i, j, k=None):
    if type(i) is int and type(j) is int and type(k) is int:
      return self.weights[i][j][k]

    return self.weights[i][j]

  def predictByX(self, X):
    # print('pred', X)
    self.X = torch.tensor((X), dtype=torch.float)
    self.predict = self.forward(self.X)

    # print('Prediction %s' % str(self.predict))
    return self.predict.data.tolist()[0][0]
