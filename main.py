import pygame as pg
import random
import sys
import os

Rect = pg.Rect

from ai import AI


class Car:
  width = 80
  height = 80

  turnSpeed = 16

  def __init__(self, screen, color):
    self.screen = screen
    self.color = color

    self.height = Car.height
    self.width = Car.width

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
    self.carRect = pg.draw.rect(self.screen, self.color,
                                (self.x, self.y, self.width, self.height))

  def draw(self):
    self.updateRects()


class Wall:

  def __init__(self, screen, color, car, score):
    self.screen = screen
    self.wallColor = color
    self.gateColor = (255, 255, 255)
    self.x = 0
    self.y = 0
    self.score = score

    # Wall Speed
    self.initSpeed = 6
    self.speedIncrease = 0.01
    self.speed = self.initSpeed

    self.carWidth = car.width
    self.width = self.screen.get_width()
    self.height = self.carWidth

    # Gate
    self.gateSizeInCarWidth = 2
    self.gateWidth = self.gateSizeInCarWidth * self.carWidth
    self.reset()

    self.score.reset()

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
    leftWallSize = random.randint(0, sizeInCarWidth - self.gateSizeInCarWidth)
    self.leftWallWidth = leftWallSize * self.carWidth

    # Left wall
    self.leftWallSurf = pg.Surface((self.leftWallWidth, self.height))
    self.leftWallSurf.fill(self.wallColor)
    self.leftWallRect = Rect(self.x, self.y, self.leftWallWidth, self.height)

    # Gate
    self.gateSurf = pg.Surface((self.gateWidth, self.height))
    self.gateSurf.fill(self.gateColor)

    # Right wall
    rightWallX = self.leftWallWidth + self.gateWidth
    self.rightWallWidth = self.width - rightWallX

    self.rightWallSurf = pg.Surface((self.rightWallWidth, self.height))
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

  def tick(self):
    self.y += self.speed
    self.updateRects()

    if self.y > self.screen.get_height():
      self.score.add()
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
    self.scoreSurface = self.font.render(str(self.score), True, (220, 0, 0))

  def add(self):
    self.score += 1
    self.renderScore()

  def draw(self):
    self.screen.blit(self.scoreSurface, (0, 0))


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

    self.fps = 60
    self.fps_clock = pg.time.Clock()

    self.isCrashed = False

    self.color = {
        'car': pg.Color(0, 150, 0),
        'background': pg.Color(200, 200, 200),
        'wall': pg.Color(50, 50, 100),
        'score': pg.Color(250, 0, 0),
        'ahead': pg.Color(100, 100, 100, 100),
        'gameOver': pg.Color(200, 50, 0)
    }

    self.font = pg.font.Font(None, 50)
    self.gameOverText = 'GAME OVER!'

    # create a surface on screen that has the size of 240 x 180
    self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT), display=1)
    self.screen.fill(self.color['background'])

    self.car = Car(self.screen, self.color['car'])
    self.score = Score(self.screen, self.color['score'], (0, 0, 30, 30))
    self.wall = Wall(self.screen, self.color['wall'], self.car, self.score)

    # For AI
    self.ai = {}

    self.ai['carAheadRect'] = Rect(self.car.x, self.car.y, self.car.width,
                                   -self.HEIGHT)
    self.ai['isAheadClean'] = True
    self.ai['position'] = 0

    # init AI and NN
    self.numberOfAIs = 1

    # TODO : 
    # 1. init AIs in list and predictions
    # 2. control car by one AI

    # Init AIs
    self.AI = []
    for i in range(self.numberOfAIs):
      W1 = [[0,0, 0], [1,0, 1]]
      W2 = [[ 0.0, 0.1], [0, 0.9], [0, 0.5]]
      weights = [W1, W2]

      layers = [2,3,2]

      self.AI.append(AI(weights, layers))

    # AIs predictions
    self.predict = []
    for i in range(self.numberOfAIs):
      self.predict.append(self.AI[i].predict([1,0.1]))

  def handleEvents(self):
    # event handling, gets all event from the event queue
    for event in pg.event.get():
      # only do something if the event is of type QUIT
      if event.type == pg.QUIT:
        # change the value to False, to exit the main loop
        pg.quit()
        sys.exit()

    # GAME events
    self.wall.tick()

    # USER events
    keys = pg.key.get_pressed()

    if keys[pg.K_LEFT]:
      self.car.left()
    if keys[pg.K_RIGHT]:
      self.car.right()

  def aiDraw(self):
    pg.draw.rect(self.screen, self.color['ahead'], self.ai['carAheadRect'], 1)

  def draw(self):
    self.screen.fill(self.color['background'])

    self.wall.draw()
    self.car.draw()
    self.score.draw()

    # Helpful info for AI
    self.aiDraw()

    if self.isCrashed:
      # GAME OVER text
      self.gameOverRender = self.font.render(
          str('%s : %d' % (self.gameOverText, self.score.score)), True,
          self.color['gameOver'])

      self.screen.blit(self.gameOverRender, (self.WIDTH // 4, self.HEIGHT // 3))

  def displayUpdate(self):
    pg.display.update()
    self.fps_clock.tick(self.fps)

    if self.isCrashed:
      # Reset all params
      self.gameOver()

  def checkCollisions(self):
    # Check collision!
    blocks = [self.wall.leftWallRect, self.wall.rightWallRect]
    self.isCrashed = self.car.carRect.collidelist(blocks) != -1

    # CALCS for AI

    # Check collision ahead car
    self.ai['carAheadRect'] = Rect(self.car.x, 0, self.car.width, self.HEIGHT)
    self.ai['isAheadClean'] = self.ai['carAheadRect'].collidelist(blocks) == -1

    self.ai['position'] = (self.car.x) / (self.WIDTH)
    print(self.car.x)

    print('ahead %s %f' % (str(self.ai['isAheadClean']), self.ai['position']))

  def gameOver(self):
    print('current score %d' % self.score.score)
    # Save current score
    self.prevAI = {
        'score': self.score.score,
        'AI': self.AI
        }

    self.AI[0].nextMutation()

    # Pause game
    pg.time.delay(1000)

    # Reset all params
    self.wall.reset()
    self.score.reset()
    self.car.reset()

  def handleAI(self):
    print('handle AI')
    # if self.ai['position'] > 0.5:
    # self.car.left()
    moreThenToTrue = 0.5

    for i in range(self.numberOfAIs):
      self.predict[i]= self.AI[i].predict([self.ai['isAheadClean'], self.ai['position']]).data.tolist()[0][0]
      print(self.predict)
      print(self.predict[i][0])

      # 0 - left
      if self.predict[i][0] > moreThenToTrue:
        print('AI turn left')
        self.car.left()
      # 1 - right
      if self.predict[i][1] > moreThenToTrue:
        print('AI turn right')
        self.car.right()


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
