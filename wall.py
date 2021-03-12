import pygame as pg
import random

Rect = pg.Rect


class Wall:

  def __init__(self, isDrawable, screen, screenWidth, color, carWidth, scores):
    self.isDrawable = isDrawable

    self.screen = screen
    self.wallColor = color
    self.gateColor = (255, 255, 255, 185)
    self.x = 0
    self.y = 0
    self.scores = scores

    self.step = 0
    self.leftWallSizesList = [1, 3, 1, 3, 1, 3, 2, 1, 2, 3]
    self.prevLeftWallSize = 0

    # Wall Speed
    self.initSpeed = 18
    self.speedIncrease = 0.00
    self.speed = self.initSpeed

    self.carWidth = carWidth
    self.width = screenWidth
    self.height = self.carWidth

    # Gate
    self.gateSizeInCarWidth = 2
    self.gateWidth = self.gateSizeInCarWidth * self.carWidth
    self.reset()

    for i in range(len(self.scores)):
      self.scores[i].reset()

  def getNextWall(self):
    self.step += 1
    self.y = -self.height
    self.speed *= (1 + self.speedIncrease)

    self.initRects()

    self.updateRects()

  def reset(self):
    self.speed = self.initSpeed
    # self.step = int(random.random()* 2)
    self.step = 0
    self.getNextWall()

  def updateRects(self):
    self.leftWallRect.y = self.y
    self.rightWallRect.y = self.y

  def getWallSize(self, sizeInCarWidth):
    if self.step < len(self.leftWallSizesList):
      return self.leftWallSizesList[self.step]

    size = random.randint(0, int(sizeInCarWidth - self.gateSizeInCarWidth))

    if size == 2:
      size -= 1

    if size == self.prevLeftWallSize:
      return self.getWallSize(sizeInCarWidth)

    return size

  def initRects(self):
    sizeInCarWidth = (self.width // self.carWidth)
    leftWallSize = self.getWallSize(sizeInCarWidth)
    self.prevLeftWallSize = leftWallSize
    self.leftWallWidth = leftWallSize * self.carWidth

    # Left wall
    self.leftWallSurf = pg.Surface((self.leftWallWidth, self.height),
                                   pg.SRCALPHA)
    self.leftWallSurf.fill(self.wallColor)
    self.leftWallRect = Rect(self.x, self.y, self.leftWallWidth, self.height)

    # Gate
    self.gateSurf = pg.Surface((self.gateWidth, self.height), pg.SRCALPHA)
    self.gateSurf.fill(self.gateColor)

    # Right wall
    rightWallX = self.leftWallWidth + self.gateWidth
    self.rightWallWidth = self.width - rightWallX

    self.rightWallSurf = pg.Surface((self.rightWallWidth, self.height),
                                    pg.SRCALPHA)
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

    # Gate center
    self.gateCenter = self.leftWallWidth + (self.gateWidth // 2)

  def tick(self, setNextScores):
    self.y += self.speed
    self.updateRects()

    if self.y > self.screen.get_height():
      setNextScores()
      self.getNextWall()

  def draw(self):
    self.screen.blit(self.wallSurface, (0, self.y))
