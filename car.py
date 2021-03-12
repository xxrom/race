import pygame as pg

Rect = pg.Rect


class Car:
  TURN_SPEED = 40

  def __init__(self, isDrawable, screen, screenWidth, carSize, color):
    self.screen = screen
    self.color = color
    self.isDrawable = False

    self.HEIGHT = carSize
    self.WIDTH = carSize
    self.TURN_SPEED = Car.TURN_SPEED
    self.SCREEN_WIDTH = screenWidth

    self.carSurf = pg.Surface((self.WIDTH, self.HEIGHT), pg.SRCALPHA)

    self.reset()
    self.tick()

  def reset(self):
    self.x = ((self.SCREEN_WIDTH / 2) // self.WIDTH) * self.WIDTH
    self.y = self.SCREEN_WIDTH - self.HEIGHT

  def right(self):
    if self.x <= self.SCREEN_WIDTH - self.WIDTH - self.TURN_SPEED:
      self.x += self.TURN_SPEED

  def left(self):
    if self.x >= self.TURN_SPEED:
      self.x -= self.TURN_SPEED

  def tick(self):
    self.updateMath()

  def updateMath(self):
    # Car rect for checking collisions
    self.carRect = Rect(self.x, self.y, self.WIDTH, self.HEIGHT)
    self.carCenter = self.x + (self.WIDTH // 2)

  def draw(self):
    # Car draw
    self.screen.blit(self.carSurf, (self.x, self.y))
    self.carSurf.fill(self.color)
