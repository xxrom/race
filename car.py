import pygame as pg

Rect = pg.Rect


class Car:
  width = 80
  height = 80

  turnSpeed = 40

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
    self.carCenter = self.x + (self.width // 2)

  def draw(self):
    self.updateRects()
