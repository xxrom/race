import random

from ai import AI


class Evolution:

  def __init__(self, numberOfChildren=5):
    self.numberOfChildren = numberOfChildren
    self.nnLayers = [2, 3, 3, 3, 3]

  def init(self):
    self.prevMaxIndex = 0

    self.population = []

    for i in range(self.numberOfChildren):
      delta = 1.0

      W0 = []
      for i in range(self.nnLayers[0]):
        w = []
        for j in range(self.nnLayers[1]):
          w.append(random.random() * delta)

        W0.append(w)

      W1 = []
      for i in range(self.nnLayers[1]):
        w = []
        for j in range(self.nnLayers[2]):
          w.append(random.random() * delta)

        W1.append(w)

      W2 = []
      for i in range(self.nnLayers[2]):
        w = []
        for j in range(self.nnLayers[3]):
          w.append(random.random() * delta)

        W2.append(w)

      W3 = []
      for i in range(self.nnLayers[3]):
        w = []
        for j in range(self.nnLayers[4]):
          w.append(random.random() * delta)

        W3.append(w)

      weights = [W0, W1, W2, W3]

      child = AI(weights, self.nnLayers)

      self.population.append(child)

  def getChildWeights(self, index):
    return self.population[index].weights

  def mutatePopulation(self, scores):
    maxScore = 0
    maxIndex = 0
    self.scores = scores

    for i in range(self.numberOfChildren):
      if self.scores[i].score >= maxScore:
        maxScore = self.scores[i].score
        maxIndex = i

    # TODO: stop car until all cars will be crashed
    print('Parent weights', self.getChildWeights(maxIndex))

    for i in range(0, self.numberOfChildren):
      if i == maxIndex or i == self.prevMaxIndex:
        continue

      self.population[i].setWeights(self.population[maxIndex].nextMutation())

  def getChildPrediciton(self, index, X):
    return self.population[index].predictByX(X)

  def getAllAIs(self):
    return self.population
