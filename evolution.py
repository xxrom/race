import random
import numpy as np

from ai import AI


class Evolution:

  def __init__(self, numberOfChildren=5):
    self.numberOfChildren = numberOfChildren
    self.nnLayers = [2, 20, 20, 2, 3]
    self.numberBestOfChildren = 2
    self.mutateRate = 0.15
    self.mutateBestChildrenRate = 0.001

    self.bestScore = 0

  def init(self):
    self.prevMaxIndex = 0

    self.population = []

    for i in range(self.numberOfChildren):
      delta = 0.1

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

      # W3 = []
      # for i in range(self.nnLayers[3]):
      # w = []
      # for j in range(self.nnLayers[4]):
      # w.append(random.random() * delta)

      # W3.append(w)

      # weights = [W0, W1, W2, W3]
      weights = [W0, W1, W2]

      child = AI(weights, self.nnLayers)

      self.population.append(child)

  def getChildWeights(self, index):
    return self.population[index].weights

  def getWeightFromBestChildren(self, i, j, k=None):
    randomChildIndex = random.randrange(0, self.numberBestOfChildren)

    if k is not None:
      return self.bestChildren[randomChildIndex].getWeightByIndexes(i, j, k)

    return self.bestChildren[randomChildIndex].getWeightByIndexes(i, j)

  def mutateChild(self, child):
    weights = []

    for i in range(len(child.weights)):

      wi = []
      for j in range(len(child.weights[i])):

        size = len(child.weights[i][j])

        wj = None
        if random.random() > self.mutateRate:
          wj = self.getWeightFromBestChildren(i, j)
        else:
          wj = np.random.random_sample(size=size)

        wi.append(wj)
      weights.append(wi)

    child.setWeights(weights)

  def mutatePopulation(self, scores):
    childrenList = []

    for i in range(self.numberOfChildren):
      childrenList.append({'index': i, 'score': scores[i].score})

    childrenList = sorted(childrenList, key=lambda x: x['score'])

    # print(childrenList)

    bestChildrenList = childrenList[-self.numberBestOfChildren:]

    # print(bestChildrenList)

    self.bestChildren = list(
        map(lambda x: self.population[x['index']], bestChildrenList))

    bestChildrenIndexes = list(map(lambda x: x['index'], bestChildrenList))
    # print(bestChildrenList, bestChildrenIndexes)

    bestScore = scores[bestChildrenIndexes[-1]].score

    if bestScore > self.bestScore:
      self.bestScore = bestScore

    print(bestChildrenIndexes)
    print(bestScore, '(', self.bestScore, ')')

    for i in range(self.numberOfChildren):
      isSkipped = i in bestChildrenIndexes and random.random(
      ) > self.mutateBestChildrenRate

      if isSkipped:
        continue

      self.mutateChild(self.population[i])

  def getChildPrediciton(self, index, X):
    return self.population[index].predictByX(X)

  def getAllAIs(self):
    return self.population
