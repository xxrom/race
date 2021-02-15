import pygame as pg
import random
import sys

rect = pg.draw.rect


class Car:
  width = 80
  height = 80

  turnSpeed = 8

  def __init__(self, surface, color):
    self.surf = surface
    self.color = color
    self.x = ((surface.get_width() / 2) // Car.width) * Car.width
    self.y = surface.get_height() - Car.height

    self.height = Car.height
    self.width = Car.width

  def right(self):
    if self.x < self.surf.get_width() - Car.width:
      self.x += Car.turnSpeed

  def left(self):
    if self.x > 0:
      self.x -= Car.turnSpeed

  def draw(self):
    rect(self.surf, self.color, (self.x, self.y, self.width, self.height))


class Wall:

  def __init__(self, surface, color, car):
    self.surf = surface
    self.color = color
    self.x = 0
    self.y = 0

    # Wall
    self.speed = 6
    self.width = self.surf.get_width()
    self.height = car.width

    # Gate
    self.gateWidth = 2 * car.width
    self.initGate()

  def initGate(self):
    self.y = -self.height
    self.gateX = random.randint(
        0, (self.width // self.gateWidth) - 1) * self.gateWidth

  def tick(self):
    self.y += self.speed

    if self.y > self.surf.get_height():
      self.initGate()

  def draw(self):
    # Wall
    rect(self.surf, self.color, (self.x, self.y, self.width, self.height))
    # Gate
    rect(self.surf, (255, 255, 255),
         (self.x + self.gateX, self.y, self.gateWidth, self.height))


class App:

  def __init__(self):
    self.HEIGHT = 680
    self.WIDTH = 400

    self.fps = 60
    self.fps_clock = pg.time.Clock()

    self.color = {
        'car': pg.Color(255, 0, 0),
        'background': pg.Color(200, 200, 200),
        'wall': pg.Color(100, 100, 100)
    }

    # initialize the pygame module
    pg.init()

    # load and set the logo
    pg.display.set_caption("Car")

    # create a surface on screen that has the size of 240 x 180
    self.screen = pg.display.set_mode((self.WIDTH, self.HEIGHT))
    self.screen.fill(self.color['background'])

    self.car = Car(self.screen, self.color['car'])
    self.wall = Wall(self.screen, self.color['wall'], self.car)

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

  def displayUpdate(self):
    pg.display.update()

    self.fps_clock.tick(self.fps)

  def run(self):
    # main loop
    while 1:

      self.handleEvents()

      self.draw()

      self.displayUpdate()


# run the main function only if this module is executed as the main script
# (if you import this as a module then nothing is executed)
if __name__ == "__main__":
  # call the main function
  app = App()
  app.run()
