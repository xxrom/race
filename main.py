import pygame as pg
import random
import sys
import os
import time
import multiprocessing

from car import Car
from wall import Wall
from score import Score
from evolution import Evolution

Rect = pg.Rect

multiprocessing.set_start_method("fork")


class App:

  def __init__(self):
    self.isDrawable = True

    # position game window in the second screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (-400, 10)

    # initialize the pygame module
    pg.init()

    # load and set the logo
    pg.display.set_caption("Car")

    # self.CPUS = 12
    # self.numberOfCars = self.CPUS * 16
    self.CPUS = 6
    self.numberOfCars = self.CPUS * 8
    self.Evolution = Evolution(self.numberOfCars)

    self.HEIGHT = 180
    self.WIDTH = 400
    self.CAR_SIZE = 80

    self.enableFps = True
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

    if len(colors) < self.numberOfCars:
      for i in range(self.numberOfCars - len(colors)):
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
    # self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), display=1)
    self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), display=0)
    self.screen.fill(self.color['background'])

    # init AI and NN
    self.Evolution.init()

    self.initTime()
    self.initCars()
    self.initScores()
    self.initGameOver()

    self.wall = Wall(self.isDrawable, self.screen, self.WIDTH,
                     self.color['wall'], self.CAR_SIZE, self.scores)

  def initTime(self):
    self.time = time.time()
    self.timeCounter = 0
    self.timeAvg = 0

  def initGameOver(self):
    self.gameOverCounter = 0

    if self.isDrawable:
      self.gameOverScore = Score(self.screen, self.color['gameOver'], (300, 20))

  def initScores(self):
    self.scores = []

    for i in range(self.numberOfCars):
      self.scores.append(Score(self.screen, self.color['car'][i], (10, i * 8)))

  def initCarPostions(self):
    self.carAheadRect = []
    self.isAheadClean = []
    self.position = []

    for i in range(self.numberOfCars):
      self.carAheadRect.append(
          Rect(self.cars[i].x, self.cars[i].y, self.cars[i].WIDTH,
               -self.HEIGHT))
      self.isAheadClean.append(True)
      self.position.append(0)

  def initCars(self):
    self.cars = []
    self.isCrashed = []

    for i in range(self.numberOfCars):
      self.cars.append(
          Car(self.isDrawable, self.screen, self.WIDTH, self.CAR_SIZE,
              self.color['car'][i]))
      self.isCrashed.append(False)

    self.initCarPostions()

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

    # USER events
    keys = pg.key.get_pressed()

    if keys[pg.K_LEFT]:
      self.cars[0].left()
    if keys[pg.K_RIGHT]:
      self.cars[0].right()

  def ticks(self):
    # GAME events
    self.wall.tick(self.setNextScores)

    for i in range(self.numberOfCars):
      if self.isCrashed[i] == True:
        continue

      self.cars[i].tick()

  def aiDraw(self):
    for i in range(self.numberOfCars):
      pg.draw.rect(self.screen, self.color['ahead'], self.carAheadRect[i], 1)

  def draw(self):
    self.screen.fill(self.color['background'])

    self.wall.draw()

    for i in range(self.numberOfCars):
      if self.isCrashed[i] == True:
        continue

      self.cars[i].draw()
      self.scores[i].draw()

    # Helpful info for AI
    self.aiDraw()

    self.gameOverScore.draw()

  def displayUpdate(self):
    if self.isDrawable:
      pg.display.update()

      if self.enableFps:
        self.fps_clock.tick(self.fps)

    isAllCarCrashed = len(list(filter(lambda x: x == True,
                                      self.isCrashed))) == len(self.isCrashed)

    if isAllCarCrashed:
      # print('isAllCrashed %s' % str(isAllCarCrashed))

      if self.isDrawable:
        # GAME OVER text
        self.gameOverRender = self.font.render(
            str('%s : %d' % (self.gameOverText, self.scores[0].score)), True,
            self.color['gameOver'])
        self.screen.blit(self.gameOverRender,
                         (self.WIDTH // 4, self.HEIGHT // 3))

      self.gameOver()

  def checkCollisions(self):
    # Check collision!
    blocks = [self.wall.leftWallRect, self.wall.rightWallRect]

    for i in range(self.numberOfCars):
      if self.isCrashed[i] == True:
        continue

      self.isCrashed[i] = self.cars[i].carRect.collidelist(blocks) != -1

      # CALCS for AI
      self.carAheadRect[i] = Rect(self.cars[i].x, 0, self.cars[i].WIDTH,
                                  self.HEIGHT)
      self.isAheadClean[i] = self.carAheadRect[i].collidelist(blocks) == -1
      self.position[i] = (self.cars[i].x) / (self.WIDTH - self.cars[i].WIDTH)

  def measureTimeExecution(self):
    self.timeCounter += 1

    currentTime = time.time()
    diff = currentTime - self.time

    self.time = currentTime
    self.timeAvg = ((self.timeAvg + diff) / 2)

    if self.timeCounter % 200 == 0:
      print('tick fps = ', 1 / self.timeAvg)

  def gameOver(self):
    self.gameOverCounter += 1

    if self.isDrawable:
      self.gameOverScore.setText(self.gameOverCounter)

    self.Evolution.mutatePopulation(self.scores)

    # Reset all params
    self.wall.reset()

    for i in range(self.numberOfCars):
      self.isCrashed[i] = False
      self.scores[i].reset()
      self.cars[i].reset()

  def handleAI(self):
    moreThenToTrue = 0.8

    def AIJob(start, end):
      for index in range(start, end):

        if self.isCrashed[index] == True:
          continue

        X = [[int(self.isAheadClean[index]), self.position[index]]]
        predict = self.Evolution.getChildPrediciton(index, X)

        if index == 0:
          print(predict)

        if predict[0] > moreThenToTrue:
          self.cars[index].left()

        if predict[1] > moreThenToTrue:
          self.cars[index].right()

        diff = abs(self.wall.gateCenter - self.cars[index].carCenter) * 0.0001

        self.scores[index].add(-diff)

    CPUS = self.CPUS

    interval = self.numberOfCars // CPUS

    jobs = []

    for i in range(CPUS):
      start = i * interval
      end = start + interval

      process = multiprocessing.Process(target=AIJob, args=(
          start,
          end,
      ))

      jobs.append(process)

    for j in jobs:
      j.start()

    for j in jobs:
      j.join()

  def run(self):

    while 1:
      # t0 = time.time()

      self.handleEvents()
      # t1 = time.time()

      self.ticks()
      # t2 = time.time()

      self.handleAI()
      # t3 = time.time()

      self.checkCollisions()
      # t4 = time.time()

      if self.isDrawable:
        self.draw()

      self.displayUpdate()
      # t5 = time.time()

      self.measureTimeExecution()
      # t6 = time.time

      # print('-------------')
      # print(t0)
      # print(t1)
      # print(t2)
      # print(t3)
      # print(t4)
      # print(t5)


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
  app = App()
  app.run()
