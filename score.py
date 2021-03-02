import pygame as pg


class Score:

  def __init__(self, screen, color, position):
    self.screen = screen
    self.color = color
    self.position = position

    self.font = pg.font.SysFont(name=None, size=16)
    self.reset()

  def reset(self):
    self.score = 0
    self.renderScore()

  def renderScore(self):
    self.scoreSurface = self.font.render(str(self.score), True, self.color)

  def add(self, addNumber=1):
    self.score += addNumber
    self.renderScore()

  def setText(self, text):
    self.score = text
    self.renderScore()

  def draw(self):
    self.screen.blit(self.scoreSurface, self.position)
