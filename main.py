import pygame as pg
import random
import sys
import os

from ai import AI
from car import Car
from wall import Wall
from score import Score

Rect = pg.Rect


class App:

  def __init__(self):
    # position game window in the second screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (-400, 10)

    # initialize the pygame module
    pg.init()

    # load and set the logo
    pg.display.set_caption("Car")

    self.numberOfAIs = 5

    self.HEIGHT = 600
    self.WIDTH = 400

    self.fps = 60
    self.fps_clock = pg.time.Clock()

    colors = [
        pg.Color(0, 150, 0, 125),
        pg.Color(150, 0, 0, 125),
        pg.Color(0, 0, 150, 125),
        pg.Color(0, 0, 100, 125),
        pg.Color(0, 100, 150, 125),
        pg.Color(100, 100, 150, 125),
        pg.Color(10, 50, 150, 125),
        pg.Color(250, 0, 150, 125),
    ]

    if len(colors) < self.numberOfAIs:
      for i in range(self.numberOfAIs - len(colors)):
        colors.append(
            pg.Color(int(random.random() * 255), int(random.random() * 255),
                     int(random.random() * 255), 125))

    self.color = {
        'car': colors,
        'background': pg.Color(230, 230, 230),
        'wall': pg.Color(50, 50, 100, 125),
        'score': colors,
        'ahead': pg.Color(100, 100, 100, 100),
        'gameOver': pg.Color(200, 50, 0)
    }

    self.font = pg.font.Font(None, 50)
    self.gameOverText = 'GAME OVER!'

    # create a surface on screen that has the size of 240 x 180
    self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), display=1)
    self.screen.fill(self.color['background'])

    self.gameOverCounter = 0
    self.gameOverScore = Score(self.screen, self.color['gameOver'], (300, 20))

    # init AI and NN
    self.initAI()
    self.initCars()
    self.initCarPostions()
    self.initScores()

    self.wall = Wall(self.screen, self.color['wall'], self.cars[0], self.scores)

  def initScores(self):
    self.scores = []

    for i in range(self.numberOfAIs):
      self.scores.append(Score(self.screen, self.color['car'][i], (10, i * 8)))

  def initCarPostions(self):
    self.carAheadRect = []
    self.isAheadClean = []
    self.position = []

    for i in range(self.numberOfAIs):
      self.carAheadRect.append(
          Rect(self.cars[i].x, self.cars[i].y, self.cars[i].width,
               -self.HEIGHT))
      self.isAheadClean.append(True)
      self.position.append(0)

  def initCars(self):
    self.cars = []
    self.isCrashed = []

    for i in range(self.numberOfAIs):
      self.cars.append(Car(self.screen, self.color['car'][i]))
      self.isCrashed.append(False)

  def initAI(self):
    self.prevMaxIndex = 0

    self.AI = []
    for i in range(self.numberOfAIs):
      layers = [2, 3, 3, 3, 3]
      delta = 1.0

      W0 = []
      for i in range(layers[0]):
        w = []
        for j in range(layers[1]):
          w.append(random.random() * delta)

        W0.append(w)

      W1 = []
      for i in range(layers[1]):
        w = []
        for j in range(layers[2]):
          w.append(random.random() * delta)

        W1.append(w)

      W2 = []
      for i in range(layers[2]):
        w = []
        for j in range(layers[3]):
          w.append(random.random() * delta)

        W2.append(w)

      W3 = []
      for i in range(layers[3]):
        w = []
        for j in range(layers[4]):
          w.append(random.random() * delta)

        W3.append(w)

      weights = [W0, W1, W2, W3]

      self.AI.append(AI(weights, layers))

    # AIs predictions
    self.predicts = []
    for i in range(self.numberOfAIs):
      self.predicts.append(self.AI[i].predictByX([1, 0.1]))

  def setNextScores(self):
    for i in range(len(self.scores)):
      if self.isCrashed[i] == False:
        self.scores[i].add()

  def handleEvents(self):
    # event handling, gets all event from the event queue
    for event in pg.event.get():
      # only do something if the event is of type QUIT
      if event.type == pg.QUIT:
        # change the value to False, to exit the main loop
        pg.quit()
        sys.exit()

    # GAME events
    self.wall.tick(self.setNextScores)

    # USER events
    # keys = pg.key.get_pressed()

    # if keys[pg.K_LEFT]:
    # self.cars[0].left()
    # if keys[pg.K_RIGHT]:
    # self.cars[0].right()

  def aiDraw(self):
    for i in range(self.numberOfAIs):
      pg.draw.rect(self.screen, self.color['ahead'], self.carAheadRect[i], 1)

  def draw(self):
    self.screen.fill(self.color['background'])

    self.wall.draw()

    for i in range(self.numberOfAIs):
      if self.isCrashed[i] == True:
        continue

      self.cars[i].draw()
      self.scores[i].draw()

    # Helpful info for AI
    # self.aiDraw()

    self.gameOverScore.draw()

  def displayUpdate(self):
    pg.display.update()
    self.fps_clock.tick(self.fps)

    isAllCarCrashed = len(list(filter(lambda x: x == True,
                                      self.isCrashed))) == len(self.isCrashed)

    if isAllCarCrashed:
      print('isAllCrashed %s' % str(isAllCarCrashed))

      # GAME OVER text
      self.gameOverRender = self.font.render(
          str('%s : %d' % (self.gameOverText, self.scores[0].score)), True,
          self.color['gameOver'])

      self.screen.blit(self.gameOverRender, (self.WIDTH // 4, self.HEIGHT // 3))

      self.gameOver()

  def checkCollisions(self):
    # Check collision!
    blocks = [self.wall.leftWallRect, self.wall.rightWallRect]

    for i in range(self.numberOfAIs):
      if self.isCrashed[i] == True:
        continue

      self.isCrashed[i] = self.cars[i].carRect.collidelist(blocks) != -1

      # CALCS for AI
      self.carAheadRect[i] = Rect(self.cars[i].x, 0, self.cars[i].width,
                                  self.HEIGHT)
      self.isAheadClean[i] = self.carAheadRect[i].collidelist(blocks) == -1
      self.position[i] = (self.cars[i].x) / (self.WIDTH - self.cars[i].width)

  def gameOver(self):
    self.gameOverCounter += 1
    self.gameOverScore.setText(self.gameOverCounter)

    maxScore = 0
    maxIndex = 0

    for i in range(self.numberOfAIs):
      if self.scores[i].score >= maxScore:
        maxScore = self.scores[i].score
        maxIndex = i

    print('>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>')
    print('GAME OVER score %d' % maxScore)
    print('>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>')

    # TODO: stop car until all cars will be crashed
    print('Parent weights', self.AI[maxIndex].weights)

    for i in range(0, self.numberOfAIs):
      if i == maxIndex or i == self.prevMaxIndex:
        continue

      if self.scores[maxIndex].score - self.scores[i].score < 1.0:
        self.AI[i].setWeights(self.AI[i].nextMutation())
      else:
        self.AI[i].setWeights(self.AI[maxIndex].nextMutation())

    self.prevMaxIndex = maxIndex

    # Reset all params
    self.wall.reset()

    for i in range(self.numberOfAIs):
      self.isCrashed[i] = False
      self.scores[i].reset()
      self.cars[i].reset()

  def handleAI(self):
    moreThenToTrue = 0.7

    for i in range(0, self.numberOfAIs):
      if self.isCrashed[i] == True:
        continue

      X = [int(self.isAheadClean[i]), self.position[i]]
      self.predicts[i] = self.AI[i].predictByX(X)

      left = False
      # 0 - left
      if self.predicts[i][0] > moreThenToTrue:
        # print('AI turn left')
        left = True

      right = False
      # 1 - right
      if self.predicts[i][1] > moreThenToTrue:
        # print('AI turn right')
        right = True

      # 2 - stop
      if left == True and right == False:
        self.cars[i].left()
      if left == False and right == True:
        self.cars[i].right()

      isShouldTurnLeft = self.wall.gateCenter - self.cars[i].carCenter < 0

      if isShouldTurnLeft == True:
        if left == True and right == False:
          self.scores[i].add(0.001)
        elif left == False and right == True:
          self.scores[i].add(0.0000001)
      else:
        if left == False and right == True:
          self.scores[i].add(0.001)
        elif left == True and right == False:
          self.scores[i].add(0.0000001)

      if (self.isAheadClean[i] == True):
        self.scores[i].add(0.01)
      else:
        self.scores[i].add(-0.000001)

  def run(self):
    while 1:
      self.handleEvents()
      self.handleAI()

      self.checkCollisions()
      self.draw()

      self.displayUpdate()


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
  # call the main function
  app = App()
  app.run()
