import pygame as pg
import random
import sys
import os

from ai import AI

Rect = pg.Rect


class Car:
  width = 80
  height = 80

  turnSpeed = 16

  def __init__(self, screen, color):
    self.screen = screen
    self.color = color

    self.height = Car.height
    self.width = Car.width

    self.carSurf = pg.Surface((self.width, self.height), pg.SRCALPHA)

    self.reset()
    self.updateRects()


  def reset(self):
    self.x = ((self.screen.get_width() / 2) // Car.width) * Car.width
    self.y = self.screen.get_height() - Car.height

  def right(self):
    if self.x <= self.screen.get_width() - Car.width - Car.turnSpeed:
      self.x += Car.turnSpeed

  def left(self):
    if self.x >= Car.turnSpeed:
      self.x -= Car.turnSpeed

  def updateRects(self):
    # Car draw
    self.screen.blit(self.carSurf, (self.x, self.y))
    self.carSurf.fill(self.color)

    # Car rect for checking collisions
    self.carRect = Rect(self.x, self.y, self.width, self.height)

  def draw(self):
    self.updateRects()

class Wall:

  def __init__(self, screen, color, car, scores):
    self.screen = screen
    self.wallColor = color
    self.gateColor = (255, 255, 255, 185)
    self.x = 0
    self.y = 0
    self.scores = scores

    # Wall Speed
    self.initSpeed = 12
    self.speedIncrease = 0.00
    self.speed = self.initSpeed

    self.carWidth = car.width
    self.width = self.screen.get_width()
    self.height = self.carWidth

    # Gate
    self.gateSizeInCarWidth = 1.5
    self.gateWidth = self.gateSizeInCarWidth * self.carWidth
    self.reset()

    for i in range(len(self.scores)):
      self.scores[i].reset()

  def getNextWall(self):
    self.y = -self.height
    self.speed *= (1 + self.speedIncrease)

    self.initRects()

    self.updateRects()

  def reset(self):
    self.speed = self.initSpeed
    self.getNextWall()

  def updateRects(self):
    self.leftWallRect.y = self.y
    self.rightWallRect.y = self.y

  def initRects(self):
    sizeInCarWidth = (self.width // self.carWidth)
    leftWallSize = random.randint(0, int(sizeInCarWidth - self.gateSizeInCarWidth))
    self.leftWallWidth = leftWallSize * self.carWidth

    # Left wall
    self.leftWallSurf = pg.Surface((self.leftWallWidth, self.height), pg.SRCALPHA)
    self.leftWallSurf.fill(self.wallColor)
    self.leftWallRect = Rect(self.x, self.y, self.leftWallWidth, self.height)

    # Gate
    self.gateSurf = pg.Surface((self.gateWidth, self.height), pg.SRCALPHA)
    self.gateSurf.fill(self.gateColor)

    # Right wall
    rightWallX = self.leftWallWidth + self.gateWidth
    self.rightWallWidth = self.width - rightWallX

    self.rightWallSurf = pg.Surface((self.rightWallWidth, self.height),pg.SRCALPHA )
    self.rightWallSurf.fill(self.wallColor)
    self.rightWallRect = Rect(self.leftWallWidth + self.gateWidth, self.y,
                              self.rightWallWidth, self.height)

    # Wall full width surface
    self.wallSurface = pg.Surface((self.width, self.height))

    # Blit all parts on full Wall surface
    self.wallSurface.blit(self.leftWallSurf, (0, 0))
    self.wallSurface.blit(self.gateSurf, (self.leftWallWidth, 0))
    self.wallSurface.blit(self.rightWallSurf,
                          (self.leftWallWidth + self.gateWidth, 0))

  def tick(self, setNextScores):
    self.y += self.speed
    self.updateRects()

    if self.y > self.screen.get_height():
      setNextScores()
      self.getNextWall()

  def draw(self):
    self.screen.blit(self.wallSurface, (0, self.y))

class Score:

  def __init__(self, screen, color, position):
    self.screen = screen
    self.color = color
    self.position = position

    self.font = pg.font.SysFont(name=None, size=30)
    self.reset()

  def reset(self):
    self.score = 0
    self.renderScore()

  def renderScore(self):
    self.scoreSurface = self.font.render(str(self.score), True, self.color)

  def add(self):
    self.score += 1
    self.renderScore()

  def draw(self):
    self.screen.blit(self.scoreSurface, self.position)

class App:

  def __init__(self):
    # position game window in the second screen
    os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (-400,10)

    # initialize the pygame module
    pg.init()

    # load and set the logo
    pg.display.set_caption("Car")

    self.HEIGHT = 680
    self.WIDTH = 400

    self.fps = 120
    self.fps_clock = pg.time.Clock()

    colors = [
        pg.Color(0, 150, 0, 125), 
        pg.Color(150, 0, 0,125), 
        pg.Color(0, 0, 150,125),
        pg.Color(0, 0, 100,125),
        pg.Color(0, 100, 150,125),
        pg.Color(100, 100, 150,125),
        pg.Color(10, 50, 150,125),
        pg.Color(0, 80, 150,125),
        pg.Color(0, 10, 150,125),
        pg.Color(10, 0, 150,125),
        pg.Color(80, 0, 150,125),
        pg.Color(250, 0, 150,125),
        pg.Color(230, 10, 150,125),
        pg.Color(240, 20, 150,125),
        pg.Color(120, 30, 150,125),
        pg.Color(130, 40, 150,125),
        pg.Color(140, 50, 150,125),
        pg.Color(50, 60, 150,125),
        pg.Color(60, 70, 150,125),
        pg.Color(80, 80, 150,125),
        pg.Color(100, 90, 150,125),
        pg.Color(80, 0, 150,125),
        pg.Color(250, 0, 150,125),
        pg.Color(230, 110, 150,125),
        pg.Color(240, 120, 150,125),
        pg.Color(120, 130, 150,125),
        pg.Color(130, 140, 150,125),
        pg.Color(140, 150, 150,125),
        pg.Color(50,160, 150,125),
        pg.Color(60, 150, 150,125),
        pg.Color(80, 180, 150,125),
        pg.Color(100, 190, 150,125),

        ]
    self.color = {
        'car': colors,
        'background': pg.Color(200, 200, 200),
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

    # init AI and NN
    self.numberOfAIs = 30

    self.initAI()
    self.initCars()
    self.initCarPostions()
    self.initScores()

    self.wall = Wall(self.screen, self.color['wall'], self.cars[0], self.scores)

  def initScores(self):
    self.scores = []

    for i in range(self.numberOfAIs):
      self.scores.append(Score(self.screen, self.color['car'][i], (10, i * 20)))

  def initCarPostions(self):
    self.carAheadRect = []
    self.isAheadClean = []
    self.position = []

    for i in range(self.numberOfAIs):
      self.carAheadRect.append(Rect(self.cars[i].x, self.cars[i].y, self.cars[i].width,
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
    self.AI = []
    for i in range(self.numberOfAIs):
      r0 = random.random()
      r1 = random.random()
      r2 = random.random()

      W1 = [[r0,r1, r2], [r2,r1, r0]]
      W2 = [[r0, r2], [r1, r2], [r1, r1]]

      weights = [W1, W2]

      layers = [2,3,2]

      self.AI.append(AI(weights, layers))

    # AIs predictions
    self.predict = []
    for i in range(self.numberOfAIs):
      self.predict.append(self.AI[i].predict([1,0.1]))

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
    keys = pg.key.get_pressed()

    if keys[pg.K_LEFT]:
      self.cars[0].left()
    if keys[pg.K_RIGHT]:
      self.cars[0].right()

  def aiDraw(self):
    for i in range(self.numberOfAIs):
      pg.draw.rect(self.screen, self.color['ahead'], self.carAheadRect[i], 1)

  def draw(self):
    self.screen.fill(self.color['background'])

    self.wall.draw()

    for i in range(self.numberOfAIs):
      self.cars[i].draw()
      self.scores[i].draw()

    # Helpful info for AI
    self.aiDraw()

  def displayUpdate(self):
    pg.display.update()
    self.fps_clock.tick(self.fps)

    isAllCarCrashed = len(list(filter(lambda x: x == True, self.isCrashed))) == len(self.isCrashed)
    
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
      if self.isCrashed[i] == False:
        self.isCrashed[i] = self.cars[i].carRect.collidelist(blocks) != -1

        # if self.isCrashed[i] == True:
          # print('%d Crashed !!!' % i)
          # pg.time.delay(300)

      # CALCS for AI
      self.carAheadRect[i] = Rect(self.cars[i].x, 0, self.cars[i].width, self.HEIGHT)
      self.isAheadClean[i] = self.carAheadRect[i].collidelist(blocks) == -1
      self.position[i] = (self.cars[i].x) / (self.WIDTH)

  def gameOver(self):
    maxScore = 0
    maxIndex = 0

    for i in range(self.numberOfAIs):
      if self.scores[i].score > maxScore:
        maxScore = self.scores[i].score
        maxIndex = i

    print('>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>')
    print('GAME OVER score %d' % maxScore) 
    print('>>>>>>>>>>>>>>>>>>>>>')
    print('>>>>>>>>>>>>>>>>>>>>>')

    # TODO: stop car until all cars will be crashed
    self.AI[0] = self.AI[maxIndex]
    self.AI[0].setWeights(self.AI[maxIndex].weights)

    for i in range(1, self.numberOfAIs):
      self.AI[i].setWeights(self.AI[0].nextMutation())

    # Pause game
    pg.time.delay(500)

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

      self.predict[i]= self.AI[i].predict([self.isAheadClean[i], self.position[i]]).data.tolist()[0][0]

      # 0 - left
      if self.predict[i][0] > moreThenToTrue:
        # print('AI turn left')
        self.cars[i].left()

      # 1 - right
      if self.predict[i][1] > moreThenToTrue:
        # print('AI turn right')
        self.cars[i].right()


  def run(self):
    # main loop
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
