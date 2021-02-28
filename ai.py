import torch
import torch.nn as nn
import random


# Neural_Network
class AI(nn.Module):

  def __init__(self, weights=None, layers=[2,3,3,3,3]):
    super(AI, self).__init__()

    self.chanceToAllNew = 0.1
    self.skipMutation = 0.8
    self.changeInterval = 0.5

    self.weights = weights.copy()
    self.layers = layers.copy()

    # parameters
    # TODO: parameters can be parameterized instead of declaring them here
    self.inputSize = layers[0]
    self.hiddenSize0 = layers[1]
    self.hiddenSize1 = layers[2]
    self.hiddenSize2 = layers[3]
    self.outputSize = layers[4]

    # weights
    if weights is not None:
      self.W0 = torch.tensor(([weights[0]]), dtype=torch.float)
      self.W1 = torch.tensor(([weights[1]]), dtype=torch.float)
      self.W2 = torch.tensor(([weights[2]]), dtype=torch.float)
      self.W3 = torch.tensor(([weights[3]]), dtype=torch.float)
    else:
      self.W0 = torch.randn(self.inputSize, self.hiddenSize0)
      self.W1 = torch.randn(self.hiddenSize0, self.hiddenSize1)
      self.W2 = torch.randn(self.hiddenSize1, self.outputSize2)
      self.W3 = torch.randn(self.hiddenSize2, self.outputSize)

    # print('>>>>>>>>>>>>>>>>>>>>>>>')
    # print('NEW NN weights', self.W0, self.W1, self.W2)
    # print('<<<<<<<<<<<<<<<<<<')

  def forward(self, X):
    # 3 X 3 ".dot" does not broadcast in PyTorch
    self.z1 = torch.matmul(X, self.W0)
    # activation function
    self.z2 = self.sigmoid(self.z1)

    self.z3 = torch.matmul(self.z2, self.W1)
    self.z4 = self.sigmoid(self.z3)

    self.z5 = torch.matmul(self.z4, self.W2)
    self.z6 = self.sigmoid(self.z5)

    self.z7 = torch.matmul(self.z6, self.W3)
    # final activation function
    o = self.sigmoid(self.z7)

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

  # def saveWeights(self, model):
    # we will use the PyTorch internal storage functions
    # torch.save(model, "NN")
    # you can reload model with all the weights and so forth with:
    # torch.load("NN")

  def setWeights(self, weights):
    self.weights = weights[:]

    self.W0 = torch.tensor(([self.weights[0]]), dtype=torch.float)
    self.W1 = torch.tensor(([self.weights[1]]), dtype=torch.float)
    self.W2 = torch.tensor(([self.weights[2]]), dtype=torch.float)
    self.W3 = torch.tensor(([self.weights[3]]), dtype=torch.float)

  def getNextWeight(self, weight):
    # skit change
    if random.random() < self.skipMutation:
      return weight

    isPlusChange = True if random.random() > 0.5  else False
    # print('isPlug %s' % str(isPlusChange))

    delta = self.changeInterval * random.random()
    # print('delta', delta)

    if random.random() > 0.5 and (weight + delta) <= 1.0:
      # print('Plus + delta', weight, weight + delta)
      return weight + delta
    elif (weight - delta) >= 0.0:
      # print('Minus - delta',weight,  weight - delta)
      return weight - delta

    return random.random()

  def nextMutation(self):
    allNew = False
    if random.random() <= self.chanceToAllNew:
      allNew = True

    weights = [] 

    for i in range(len(self.weights)):

      wi = []
      for j in range(len(self.weights[i])):

        wj = []
        for k in range(len(self.weights[i][j])):
          if allNew == False:
            wj.append(self.getNextWeight(self.weights[i][j][k]))
          else:
            wj.append(random.random())
          # print('Old w = %f => %f' % (self.weights[i][j][k], wj[len(wj) -1 ]))

        wi.append(wj)
      weights.append(wi)

    return weights

  def predictByX(self, X):
    # print('pred', X)
    self.X = torch.tensor((X), dtype=torch.float)
    self.predict = self.forward(self.X)

    # print('Prediction %s' % str(self.predict)) 
    return self.predict.data.tolist()[0][0]

