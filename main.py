import pygame as pg
import random
import sys

Rect = pg.Rect


class Car:
  width = 80
  height = 80

  turnSpeed = 14

  def __init__(self, surface, color):
    self.surface = surface
    self.color = color
    self.x = 0
    self.y = 0

    self.height = Car.height
    self.width = Car.width

    self.reset()
    self.updateRects()

  def reset(self):
    self.x = ((self.surface.get_width() / 2) // Car.width) * Car.width
    self.y = self.surface.get_height() - Car.height

  def right(self):
    if self.x < self.surface.get_width() - Car.width:
      self.x += Car.turnSpeed

  def left(self):
    if self.x > 0:
      self.x -= Car.turnSpeed

  def updateRects(self):
    self.carRect = pg.draw.rect(self.surface, self.color,
                                (self.x, self.y, self.width, self.height))

  def draw(self):
    self.updateRects()


class Wall:

  def __init__(self, surface, color, car, score):
    self.surface = surface
    self.wallColor = color
    self.gateColor = (255, 255, 255)
    self.x = 0
    self.y = 0
    self.score = score

    # Wall Speed
    self.initSpeed = 6
    self.speedIncrease = 0.5
    self.speed = self.initSpeed

    self.carWidth = car.width
    self.width = self.surface.get_width()
    self.height = self.carWidth

    # Gate
    self.gateSizeInCarWidth = 2
    self.gateWidth = self.gateSizeInCarWidth * self.carWidth
    self.reset()

    self.score.reset()

  def getNextWall(self):
    self.y = -self.height
    self.speed += self.speedIncrease

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

    if self.y > self.surface.get_height():
      self.score.add()
      self.getNextWall()

  def draw(self):
    self.surface.blit(self.wallSurface, (0, self.y))


class Score:

  def __init__(self, serface, color, position):
    self.serface = serface
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
    self.serface.blit(self.scoreSurface, (0, 0))


class App:

  def __init__(self):
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
        'car': pg.Color(255, 0, 0),
        'background': pg.Color(200, 200, 200),
        'wall': pg.Color(100, 100, 100),
        'score': pg.Color(250, 0, 0),
        'gameOver': pg.Color(200, 50, 0)
    }

    self.font = pg.font.Font(None, 50)
    self.gameOverText = 'GAME OVER!'

    # create a surface on screen that has the size of 240 x 180
    self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
    self.screen.fill(self.color['background'])

    self.car = Car(self.screen, self.color['car'])
    self.score = Score(self.screen, self.color['score'], (0, 0, 30, 30))
    self.wall = Wall(self.screen, self.color['wall'], self.car, self.score)

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

  def draw(self):
    self.screen.fill(self.color['background'])

    self.wall.draw()
    self.car.draw()
    self.score.draw()

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
    self.isCrashed = self.car.carRect.collidelist(
        [self.wall.leftWallRect, self.wall.rightWallRect]) != -1

  def gameOver(self):
    # Pause game
    pg.time.delay(1000)

    # Reset all params
    self.wall.reset()
    self.score.reset()
    self.car.reset()

  def run(self):
    # main loop
    while 1:
      self.handleEvents()

      self.checkCollisions()
      self.draw()

      self.displayUpdate()


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
  # call the main function
  app = App()
  app.run()
