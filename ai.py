import torch
import torch.nn as nn
import random


class Neural_Network(nn.Module):

  def __init__(self, weights=None, layers=[2,3,1]):
    super(Neural_Network, self).__init__()
    # parameters
    # TODO: parameters can be parameterized instead of declaring them here
    self.inputSize = layers[0]
    self.hiddenSize = layers[1]
    self.outputSize = layers[2]

    # weights
    if weights is not None:
      self.W1 = weights[0]
      self.W2 = weights[1]
    else:
      # 2 X 3 tensor
      self.W1 = torch.randn(self.inputSize, self.hiddenSize)
      # 3 X 1 tensor
      self.W2 = torch.randn(self.hiddenSize, self.outputSize)

    print(self.W1, self.W2)

  def forward(self, X):
    # 3 X 3 ".dot" does not broadcast in PyTorch
    self.z = torch.matmul(X, self.W1)
    # activation function
    self.z2 = self.sigmoid(self.z)
    self.z3 = torch.matmul(self.z2, self.W2)
    # final activation function
    o = self.sigmoid(self.z3)

    return o

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

  def saveWeights(self, model):
    # we will use the PyTorch internal storage functions
    torch.save(model, "NN")
    # you can reload model with all the weights and so forth with:
    # torch.load("NN")

  def predict(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): %s \n" % (str(self.xPredicted)))


class AI:

  def __init__(self, weights, layers):
    self.weights = weights

    # 2 x 3 (2 input to 3 hidden nodes)
    # self.W1 = torch.tensor(([[1, 1,0.4], [1,1,0.1]]), dtype=torch.float)
    self.W1 = torch.tensor(([self.weights[0]]), dtype=torch.float)

    # 3 x 1 (3 hidden to 1 output)
    # self.W2 = torch.tensor(([[0.1], [1], [1]]), dtype=torch.float)
    self.W2 = torch.tensor(([self.weights[1]]), dtype=torch.float)

    self.NN = Neural_Network([self.W1, self.W2], layers)


  def setWeights(self, weights):
    self.W1 = torch.tensor(([self.weights[0]]), dtype=torch.float)
    self.W2 = torch.tensor(([self.weights[1]]), dtype=torch.float)

    self.NN = Neural_Network([self.W1, self.W2])

  def getWeights(self):
    return self.weights

  def getNextWeight(self, weight):
    oldWeight = weight

    change = 0.1

    isPlusChange = True if random.random() > 0.5  else False

    delta = weight * change * random.random()

    if isPlusChange and (weight + delta) <= 1.0:
      weight += delta
    elif (weight - delta) >= 0.0:
      weight -= delta

    print('oldW %f => newW %f' % (oldWeight, weight))

    return weight


  def nextMutation(self):
    print(self.weights)
    print(len(self.weights))

    for i in range(len(self.weights)):
      for j in range(len(self.weights[i])):
        for k in range(len(self.weights[i][j])):
          self.weights[i][j][k] = self.getNextWeight(self.weights[i][j][k])

    self.setWeights(self.weights)

  def predict(self, X):
    self.X = torch.tensor(([X]), dtype=torch.float)
    predict = self.NN.forward(self.X)

    print('Prediction %s' % str(predict))

    return predict
