#
# Jam Asteroids
#
# Author : yuiio@sotodesign.org

# History
# - 2022-06-28
# Update to run with pyxel v1.7.1 (thx [kra53n](https://github.com/kra53n))

# - 2021-04-13
# Post game jam version, adding :


# Gameplay :
#   * General rebalancing
#   * More easy at start, difficulties appear more gradually
#   * Only five full levels
#   * Add bonus ovni flying saucer
#   * Countdown score to previous level when retrying a level

# UI :
#   * Add retry screen
#   * Add skipable countdown for screens
#   * Add 5 best hiscores table

# Gfx :
#   * Add flash and implosion when dying
#   * Add points bonus

# Sound :
#   * New ambiance, progressive rythme augmentation
#   * Flying saucer alarm
#   * Music for victory ending


import os
import math
import random
from enum import Enum
from time import time
import pyxel as px


# Game
SCREEN_W = 128
SCREEN_H = 128
COUNTDOWN = 10  # in seconds
GAME_SCALE = 4  # 4
BULLETS_PER_SECOND = 4

# Pyxel
SHOW_CURSOR = False  # mouse cursor visibility
## sounds
CHAN_FIRE = 0  # Ship bullet
CHAN_MAIN = 1  # play sound ambiance and death ship
CHAN_DESTROY = 2  # Explosion of asteroid and ovni hit
CHAN_THRUST = 3  # Sound engine

# Colors
DARK_BLUE = 1
PURPLE = 2
WHITE = 7
RED = 8
GREEN = 11
CYAN = 12
GREY = 13

# Math
PI = math.pi
DOUBLE_PI = PI * 2
HALF_PI = PI / 2


class STATE(Enum):
    title = 0
    newlevel = 1
    play = 2
    retry = 3
    downscore = 4
    win = 5
    gameover = 6


def center_msg_at_line(msg, line, color):
    # msg : string to output
    # line : between 1  and 16
    # color
    x = (SCREEN_W - (len(msg) * 8 / 2)) / 2
    y = line * 8 + 1
    px.text(x, y, msg, color)


def get_center(points):
    xs, ys = zip(*points)
    return V2((max(xs) - min(xs)) / 2, (max(ys) - min(ys)) / 2)


def get_radius(points, center):
    return max([V2.dist_between(p, center) for p in points])


def bound_screen(point, radius):
    if point.x + radius < 0:
        point.x = SCREEN_W + radius
    elif point.x >= SCREEN_W + radius:
        point.x = -radius
    if point.y + radius < 0:
        point.y = SCREEN_H + radius
    elif point.y - radius >= SCREEN_H:
        point.y = -radius
    return point


def is_out_screen(point, radius):
    if (
        point.x + radius < 0
        or point.x >= SCREEN_W + radius
        or point.y + radius < 0
        or point.y - radius >= SCREEN_H
    ):
        return True
    return False


def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return V2(qx, qy)


def are_collided(obj1, obj2):
    #
    #     if obj1.radius + obj2.radius >= V2.dist_between(obj1.pos, obj2.pos):
    #         return True
    #     return False
    #
    # Optimisation that get ride of math.sqrt use in dist_between
    dist = obj2.pos - obj1.pos
    dist_square = dist.x * dist.x + dist.y * dist.y
    return (obj1.radius + obj2.radius) * (obj1.radius + obj2.radius) >= dist_square


class V2:
    """A simple 2d vector class"""

    def __init__(self, *args):
        """ Create a vector, example: v = V2(1,2) """
        if len(args) == 0:
            self.values = (0, 0)
        else:
            self.values = args

    @property
    def x(self):
        return self.values[0]

    @x.setter
    def x(self, value):
        x = value
        y = self.values[1]
        self.values = (x, y)

    @property
    def y(self):
        return self.values[1]

    @y.setter
    def y(self, value):
        y = value
        x = self.values[0]
        self.values = (x, y)

    @staticmethod
    def angle_between(v1, v2):
        """Returns the angle in radians between two vectors.
        Example : d = Vector.angle_between(Vector(1,2), Vector(3,4))"""
        v1 = v1.normalize()
        v2 = v2.normalize()
        return math.acos(v1.inner(v2))

    @staticmethod
    def dist_between(v1, v2):
        """ Returns the distance betweeb two vector) """
        dist = v2 - v1
        return dist.norm()

    def copy(self):
        return V2(*self.values)

    def norm(self):
        """ Returns the norm (length, magnitude) of the vector """
        return math.sqrt(sum(comp ** 2 for comp in self))

    def argument(self):
        """ Returns the argument of the vector, the angle clockwise from +y."""
        arg_in_rad = math.acos(V2(0, 1) * self / self.norm())
        arg_in_deg = math.degrees(arg_in_rad)
        if self.values[0] < 0:
            return 360 - arg_in_deg
        else:
            return arg_in_deg

    def normalize(self):
        """ Returns a normalized unit vector """
        norm = self.norm()
        normed = tuple(comp / norm for comp in self)
        return V2(*normed)

    def rotate(self, *args):
        """Rotate this vector. If passed a number, assumes this is a
        2D vector and rotates by the passed value in degrees.  Otherwise,
        assumes the passed value is a list acting as a matrix which rotates the vector.
        """
        if len(args) == 1 and type(args[0]) == type(1) or type(args[0]) == type(1.0):
            # So, if rotate is passed an int or a float...
            if len(self) != 2:
                raise ValueError("Rotation axis not defined for greater than 2D vector")
            return self._rotate2D(*args)
        elif len(args) == 1:
            matrix = args[0]
            if not all(len(row) == len(v) for row in matrix) or not len(matrix) == len(
                self
            ):
                raise ValueError(
                    "Rotation matrix must be square and same dimensions as vector"
                )
            return self.matrix_mult(matrix)

    def _rotate2D(self, theta):
        """Rotate this vector by theta in degrees.

        Returns a new vector.
        """
        theta = math.radians(theta)
        # Just applying the 2D rotation matrix
        dc, ds = math.cos(theta), math.sin(theta)
        x, y = self.values
        x, y = dc * x - ds * y, ds * x + dc * y
        return V2(x, y)

    def inner(self, other):
        """Returns the dot product (inner product) of self and other vector"""
        return sum(a * b for a, b in zip(self, other))

    def __mul__(self, other):
        """Returns the dot product of self and other if multiplied
        by another Vector.  If multiplied by an int or float,
        multiplies each component by other.
        """
        if type(other) == type(self):
            return self.inner(other)
        elif type(other) == type(1) or type(other) == type(1.0):
            product = tuple(a * other for a in self)
            return V2(*product)

    def __rmul__(self, other):
        """ Called if 4*self for instance """
        return self.__mul__(other)

    def __div__(self, other):
        if type(other) == type(1) or type(other) == type(1.0):
            divided = tuple(a / other for a in self)
            return V2(*divided)

    def __add__(self, other):
        """ Returns the vector addition of self and other """
        added = tuple(a + b for a, b in zip(self, other))
        return V2(*added)

    def __sub__(self, other):
        """ Returns the vector difference of self and other """
        subbed = tuple(a - b for a, b in zip(self, other))
        return V2(*subbed)

    def __iter__(self):
        return self.values.__iter__()

    def __len__(self):
        return len(self.values)

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self):
        return str(self.values)


class Delay:
    def __init__(self, func, delay=0):
        self.start = time() + delay
        self.to_run = func
        self.over = False
        self.running = False

    def update(self, dt):
        t = time()
        if t >= self.start and not self.running:
            self.running = True
        if self.running:
            self.to_run.update(dt)
            if self.to_run.over:
                self.over = True
                self.running = False

    def draw(self):
        if self.running:
            self.to_run.draw()


class BonusFx:
    def __init__(self, pos, msg):
        self.pos = pos
        self.lifespan = self.duration = 1  # seconds
        self.over = False
        self.msg = msg

    def update(self, dt):
        self.lifespan -= dt
        if self.lifespan <= 0:
            self.over = True

    def draw(self):
        color = WHITE
        if self.lifespan < self.duration * 0.125:
            color = DARK_BLUE
        elif self.lifespan < self.duration * 0.25:
            color = GREY

        pad_dim = 9 / self.duration  # max pad  = 8
        msg_len = len(self.msg)
        for i, letter in enumerate(self.msg):
            pad = pad_dim * (self.duration - self.lifespan)
            msg_padded = msg_len * pad
            x = self.pos.x - (msg_padded / 2) + (i * pad)
            y = self.pos.y
            px.text(x, y, letter, color)


class Particle:

    sizes = {1: V2(1, 1), 2: V2(0, 2), 3: V2(2, 2)}

    def __init__(self, pos, size, vel, acc, duration, color):
        self.pos = pos
        self.radius = size
        self.vel = vel
        self.acc = acc
        self.lifespan = self.duration = duration
        self.color = color

    @property
    def is_dead(self):
        return True if self.lifespan <= 0 else False

    def update(self, dt):
        self.lifespan -= dt
        self.vel += self.acc
        self.pos += self.vel * dt

    def draw(self):
        pos = self.pos
        dim = Particle.sizes[self.radius]
        px.rect(*pos, *dim, self.color)


class ScreenFlash:
    def draw(self):
        px.rect(0, 0, SCREEN_W, SCREEN_H, WHITE)
        self.over = True


class ParticleImplosion:
    def __init__(self, pos, color=GREY, duration=0.5, acceleration=5):
        self.particles = []
        self.over = False
        for _ in range(30):
            size = 1
            dist = random.randint(30, 40)
            vel = V2(-dist, 0)
            vel = vel.rotate(random.uniform(0, 360))
            acc = vel.normalize() * acceleration
            ppos = pos - vel
            self.particles.append(Particle(ppos, size, vel, acc, duration, color))

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
            if p.is_dead:
                self.particles.remove(p)
                del p
        if len(self.particles) == 0:
            self.over = True

    def draw(self):
        for p in self.particles:
            p.draw()


class ParticlesExplosion:
    def __init__(self, pos, color=GREY, duration=0.5, acceleration=-0.5):
        self.particles = []
        self.over = False
        for _ in range(40):
            size = random.randint(1, 3)
            dist = random.randint(60, 80)  # distance per seconds
            vel = V2(dist, 0)
            vel = vel.rotate(random.uniform(0, 360))
            acc = vel.normalize() * acceleration  # deceleration
            self.particles.append(Particle(pos, size, vel, acc, duration, color))

    def update(self, dt):
        for p in self.particles:
            p.update(dt)
            if p.is_dead:
                self.particles.remove(p)
                del p
        if len(self.particles) == 0:
            self.over = True

    def draw(self):
        for p in self.particles:
            p.draw()


class BigExplosion:
    def __init__(self, pos):
        self.explosions = [
            ParticlesExplosion(pos, duration=1, acceleration=2, color=PURPLE),
            ParticlesExplosion(pos, duration=1, acceleration=2, color=DARK_BLUE),
            ParticlesExplosion(pos, duration=3, acceleration=-0.35, color=WHITE),
            ParticlesExplosion(pos, duration=3, acceleration=-0.35, color=CYAN),
        ]
        self.over = False

    def update(self, dt):
        for e in self.explosions:
            e.update(dt)
            if len(e.particles) == 0:
                self.explosions.remove(e)
        if len(self.explosions) == 0:
            self.over = True

    def draw(self):
        for e in self.explosions:
            e.draw()


class TurboJetEngine:
    def __init__(self, pos):
        self.particles = []
        self.pos = pos

    def add_particle(self, pos, vel):
        acc = V2()
        vel += V2(random.uniform(-2, 2), random.uniform(-2, 2))
        duration = 1.4

        self.particles.append(Particle(pos, 1, vel, acc, duration, WHITE))

    def update(self, dt, pos, thrust, thrusting):
        self.pos = pos
        if thrusting:
            self.add_particle(pos, thrust)

        for p in self.particles:
            p.update(dt)
            if p.is_dead:
                self.particles.remove(p)
                del p

    def draw(self):
        for p in self.particles:
            if p.lifespan < 1:
                p.color = DARK_BLUE
            p.draw()


class Spaceship:
    _datas = [
        V2(0, 0),
        V2(1, 1),  # engine pos
        V2(0, 2),
        V2(3, 1),
    ]

    def __init__(self, pos):

        self.load_at(GAME_SCALE)
        self.points = [None for i in range(len(self._points))]  # points for drawing

        self.orientation = PI + HALF_PI
        self.pos = pos  # - self.center  # as v2

        self.speed_turn = 0.15  # angle in radian
        self.vel = V2()
        # self.acc = V2()
        self.power = 1.8
        self.reloaded = 1

        self.update_points()
        self.jet = TurboJetEngine(self.points[1])

    def load_at(self, size):
        self._points = list(map(lambda x: x * size, self._datas))
        self.center = get_center(self._points)
        self.radius = get_radius(self._points, self.center) * 0.55

    def update_points(self):
        for i, point in enumerate(self._points):
            self.points[i] = rotate(self.center, point, self.orientation)
            self.points[i] += self.pos - self.center

    def get_direction(self, angle):
        return V2(math.cos(angle), math.sin(angle))

    def update(self, dt):

        # Change spaceship orientation
        if px.btnp(px.KEY_LEFT, 1, 1):
            self.orientation -= self.speed_turn
            if self.orientation < 0:
                self.orientation = DOUBLE_PI
        elif px.btnp(px.KEY_RIGHT, 1, 1):
            self.orientation += self.speed_turn
            if self.orientation > DOUBLE_PI:
                self.orientation = 0

        # Thrust
        thrust = V2()
        thrusting = False

        if px.btnp(px.KEY_UP, 1, 1):
            thrust = self.get_direction(self.orientation)
            thrusting = True
            px.play(CHAN_THRUST, 17)

        # Set position
        self.vel += thrust * self.power
        self.pos += self.vel * dt

        # Screen bounds
        bound_screen(self.pos, self.radius)

        # Update all points for drawing
        self.update_points()

        # jet engin
        self.jet.update(dt, self.points[1], thrust * -10, thrusting)

        # Fire
        if self.reloaded < 1:
            self.reloaded = min(1, self.reloaded + dt * BULLETS_PER_SECOND)
        if px.btnp(px.KEY_SPACE, 1, 1) and self.reloaded == 1:
            self.reloaded = 0
            Bullet(self.points[3], self.get_direction(self.orientation))
            px.play(CHAN_FIRE, 4)

    def draw(self):
        self.jet.draw()
        for p in range(1, len(self.points)):
            px.line(*self.points[p - 1], *self.points[p], WHITE)
        px.line(*self.points[p], *self.points[0], WHITE)


class Bullet:

    bullets = []

    def __init__(self, pos, orientation):
        self.pos = pos
        self.radius = 0.5
        self.vel = orientation * 50
        self.bullets.append(self)

    def update(self, dt):
        self.pos += self.vel * dt

        if (
            self.pos.x < 0
            or self.pos.x >= SCREEN_W
            or self.pos.y < 0
            or self.pos.y >= SCREEN_H
        ):
            self.die()

    def draw(self):
        px.pset(*self.pos, GREEN)

    def hit(self):
        self.die()

    def die(self):
        if self in self.bullets:
            self.bullets.remove(self)


class OvniBullet(Bullet):
    bullets = []

    def __init__(self, pos, orientation):
        self.pos = pos
        self.radius = 0.5
        self.vel = orientation * 50
        self.bullets.append(self)

    def draw(self):
        px.pset(*self.pos, RED)


class Asteroid:

    # 6 shapes ( from small to big )
    _datas = [
        [V2(0.5, 0), V2(0, 0.5), V2(0.5, 1), V2(1, 1), V2(1, 0.5)],
        [V2(0, 0), V2(0, 1), V2(0.5, 2), V2(2, 1.5), V2(1.5, 0)],
        [V2(1.5, 0), V2(0, 1), V2(0, 2), V2(1, 3), V2(2, 2), V2(2.5, 0.5)],
        [
            V2(2.5, 0),
            V2(0, 1.5),
            V2(1, 3.5),
            V2(3, 4.5),
            V2(3.5, 3.5),
            V2(4.5, 3),
            V2(4.5, 1),
        ],
        [
            V2(2.5, 0),
            V2(0, 2),
            V2(1.5, 5),
            V2(0.5, 6),
            V2(3, 8),
            V2(8, 6),
            V2(7, 2),
        ],
        [
            V2(7, 0),
            V2(2, 3),
            V2(0, 11),
            V2(7, 13),
            V2(13, 11),
            V2(12, 5),
            V2(10, 4),
        ],
    ]

    asteroids = []

    def __init__(self, pos, size=3):
        self.pos = pos
        self.size = size
        self.load_at(7 - size)

        self.points = [
            None for i in range(len(self._points[self.size]))
        ]  # points for drawing

        dist = random.randint(10, 30 - (3 * self.size))  # distance per seconds
        self.vel = V2(dist, 0)
        self.vel = self.vel.rotate(random.uniform(0, 360))

        self.orientation = random.uniform(0, DOUBLE_PI)
        self.speed_rotation = random.uniform(-2, 2)

        self.asteroids.append(self)

    def load_at(self, size):
        self._points = []
        for data in self._datas:
            self._points.append(list(map(lambda x: x * size, data)))
        self.center = get_center(self._points[self.size])
        self.radius = get_radius(self._points[self.size], self.center) * 0.8

    def update_points(self):
        for i, point in enumerate(self._points[self.size]):
            self.points[i] = rotate(self.center, point, self.orientation)
            self.points[i] += self.pos - self.center

    def update(self, dt):
        self.orientation += self.speed_rotation * dt
        self.pos += self.vel * dt
        bound_screen(self.pos, self.radius)

        # Update all points for drawing
        self.update_points()

    def draw(self):
        if not None in self.points:
            for p in range(1, len(self.points)):
                px.line(*self.points[p - 1], *self.points[p], GREY)
            px.line(*self.points[p], *self.points[0], GREY)

    def hit(self):
        if self.size > 0:
            for i in range(2):
                Asteroid(self.pos, self.size - 1)
        self.die()

    def die(self):
        if self in self.asteroids:
            self.asteroids.remove(self)


class Ovni:

    ovnis = []
    _datas = [
        [
            V2(0, 1.75),
            V2(1.5, 1),
            V2(3, 1.25),
            V2(4.5, 1),
            V2(6, 1.75),
            V2(3, 3),
            V2(0, 1.75),
        ],
        [
            V2(1.5, 1),
            V2(2, 0.25),
            V2(3, 0),
            V2(4, 0.25),
            V2(4.5, 1),
        ],
    ]

    def __init__(self, size=3):

        self.load_at(size)  # size = 3 - (level-1) * 0.5
        self.lines = [[None] * 7, [None] * 5]

        self.orientation = 0  # angle in radian
        self.speed_turn = 0.025  # angle in radian
        self.pos = V2()
        self.vel = V2(20, 20)
        self.update_points()

        self.reloaded = 1
        self.change_direction = 0
        self.life_span = 15
        self.can_fire = True

        # Alarm ovni sound; will be stopped when ovni die
        px.play(CHAN_MAIN, 22, loop=True)

        self.ovnis.append(self)

    def load_at(self, size):
        self._lines = []
        points = []
        for d in self._datas:
            line = list(map(lambda x: x * size, d))
            self._lines.append(line)
            points.extend(line)
        self.center = get_center(points)
        self.radius = get_radius(points, self.center) * 0.55

    def update_points(self):
        for i, l in enumerate(self._lines):
            for j, point in enumerate(l):
                self.lines[i][j] = rotate(self.center, point, self.orientation)
                self.lines[i][j] += self.pos - self.center

    def update(self, dt, ship, state_game):

        self.life_span -= dt
        is_alive = self.life_span > 0

        # Set position
        self.pos += self.vel * dt

        # Oscillation
        self.orientation += self.speed_turn
        if abs(self.orientation) > 0.2:
            self.speed_turn *= -1

        if is_alive:
            # Change direction
            if self.change_direction < 1:
                self.change_direction = min(
                    1, self.change_direction + dt / 3
                )  # every 3 seconds
            if self.change_direction == 1:
                self.change_direction = 0
                angle = random.randint(30, 45)
                self.vel = self.vel.rotate(random.choice((-angle, angle)))

            # Screen bounds
            bound_screen(self.pos, self.radius)

        elif is_out_screen(
            self.pos, self.radius
        ):  # Let's ovni get out of screen before dying
            self.die()

        # Update all points for drawing
        self.update_points()

        # Fire
        if self.reloaded < 1:
            bullet_per_sec = 0.75
            self.reloaded = min(1, self.reloaded + dt * bullet_per_sec)
        if self.reloaded == 1 and state_game == STATE.play and self.can_fire:
            self.reloaded = 0

            if ship.pos.y < self.pos.y:
                if ship.pos.x < self.pos.x:
                    bullet_pos = self.lines[0][0]  # ovin left
                else:
                    bullet_pos = self.lines[0][4]  # ovni rigth
            else:
                bullet_pos = self.lines[0][5]  # ovni bottom
            bullet_orientation = (ship.pos - bullet_pos).normalize()
            OvniBullet(
                bullet_pos,
                bullet_orientation,
            )
            px.play(CHAN_FIRE, 0)

    def draw(self):
        for line in self.lines:
            for p in range(1, len(line)):
                px.line(*line[p - 1], *line[p], WHITE)

    def hit(self):
        self.die()

    def die(self):
        # Stop Alarm and restore sound
        px.sound(12).speed = 30
        px.play(CHAN_MAIN, 12, loop=True)
        # Kill ovni
        if self in self.ovnis:
            self.ovnis.remove(self)


class App:

    gfx = []
    level_msg = [
        "Training zone",
        "A little warmup",
        "Let's move",
        "Rock'n roll baby",
        "What the hell ?",
        "Final step",
    ]
    highscores = [15000, 11250, 7500, 3750, 0000]  # default hi-scores

    def __init__(self):
        px.init(SCREEN_W, SCREEN_H, title="Asteroid")
        px.fullscreen(True)
        px.load(os.path.join("assets", "asteroid.pyxres"))

        self.pt = time()  # Buffer previous time
        self.ship = Spaceship(V2(SCREEN_W / 2, SCREEN_H / 2))
        self.change_state(STATE.title)

        px.mouse(SHOW_CURSOR)
        px.run(self.update, self.draw)

    def create_asteroids(self, level):
        asteroid_size = min(5, level + 1)  # can be 0 to 5
        asteroid_number = min(2, level) + level // 5  # 1,2,2,2,3
        for i in range(asteroid_number):
            Asteroid(
                V2(
                    random.uniform(-10, 10) + random.randint(0, 1) * SCREEN_W,
                    random.uniform(-10, 10) + random.randint(0, 1) * SCREEN_H,
                ),
                asteroid_size,
            )

    def center_spaceship(self):
        self.ship.load_at(6 - self.level)  # scale
        self.ship.pos = V2(SCREEN_W / 2, SCREEN_H / 2)
        self.ship.vel = V2()
        self.ship.jet.particles.clear()
        self.ship.orientation = PI + HALF_PI
        self.ship.update_points()
        self.power = 1.8 - (self.level - 1) * 0.075  # speed : 1.8 to 1.5

    def get_ovni_points(self):
        return 5000 + (self.level - 1) * 2500

    def ovni_death(self, ovni):
        ovni_points = self.get_ovni_points()
        self.gfx.append(ParticlesExplosion(ovni.pos.copy()))
        self.gfx.append(BonusFx(ovni.pos.copy(), str(ovni_points)))
        px.play(CHAN_DESTROY, random.randint(1, 3))
        ovni.hit()
        self.score += ovni_points
        self.score_level += ovni_points

    def ship_death(self):
        self.gfx.append(ScreenFlash())
        self.gfx.append(ParticleImplosion(self.ship.pos))
        self.gfx.append(Delay(BigExplosion(self.ship.pos), delay=1.25))
        px.stop(1)
        px.play(CHAN_MAIN, 5)
        self.change_state(STATE.retry)

    def asteroid_death(self, asteroid):
        self.gfx.append(ParticlesExplosion(asteroid.pos.copy()))
        px.play(CHAN_DESTROY, random.randint(1, 3))
        asteroid.hit()
        asteroid_points = 200 - asteroid.size * 10
        self.score += asteroid_points
        self.score_level += asteroid_points

    def ending_gfx(self):
        # Fireworks for victor ending
        for _ in range(15):
            delay = random.uniform(0, 5)
            pos = V2(
                random.randint(20, SCREEN_W - 20), random.randint(20, SCREEN_H - 20)
            )
            self.gfx.append(Delay(ParticlesExplosion(pos), delay=delay))

        for _ in range(5):
            delay = random.uniform(0, 3)
            pos = V2(
                random.randint(20, SCREEN_W - 20), random.randint(20, SCREEN_H - 20)
            )
            self.gfx.append(Delay(BigExplosion(pos), delay=delay))

    def is_countdown_ended(self):
        # - To start a countdown : self.start_countdown = time() somewhere
        # - Constant COUNTDOWN set the duration in seconds
        # - And use this function to know when the countdown is over
        self.countdown = COUNTDOWN - 1 - int(time() - self.start_countdown)
        if self.countdown < 0:
            return True
        return False

    def print_highscores_at_line(self, start_line):
        # print high scores table
        max_size = 6
        center_msg_at_line("HIGH SCORES", start_line, GREY)
        for i in range(5):
            color = DARK_BLUE
            if self.highscores[i] == self.score:
                color = GREY
                x = (SCREEN_W - (max_size * 5)) // 2 - 1
                y = (i + start_line + 1) * 8 + 1
                px.text(x, y, ">", color)
            center_msg_at_line(
                str(self.highscores[i]).zfill(max_size),
                start_line + 1 + i,
                color,
            )

    def change_state(self, state):

        # GAME LOGIC

        previous_state = self.state if hasattr(self, "state") else -1
        self.state = state

        # When retry at the end of down score
        if previous_state == STATE.downscore and self.state == STATE.newlevel:
            self.level -= 1  # To keep same level when enter STATE.newlevel

        # Opening
        if self.state == STATE.title:
            self.score = 0
            self.score_level = 0
            self.level = 0
            self.create_asteroids(3)  # Decoration only for eyes at start
            px.stop()  # Stop all sounds

        if self.state == STATE.newlevel:

            if self.level == 0:  # if coming from opening ...
                Asteroid.asteroids.clear()  # ... clear decoration

            self.level += 1

            px.sound(12).speed = max(
                13, px.sound(12).speed
            )  # add a minimum delay for respawn safely
            px.play(CHAN_MAIN, 12, loop=True)

            if self.level > 5:
                self.state = STATE.win
            else:
                if previous_state == STATE.downscore:
                    Asteroid.asteroids.clear()
                # Let'start a new level
                self.score_level = 0
                self.center_spaceship()
                self.create_asteroids(self.level)
                self.countdown = COUNTDOWN
                self.start_countdown = time()

        if self.state == STATE.retry:
            for ovni in Ovni.ovnis:
                ovni.life_span = 0
                ovni.can_fire = False

            self.start_countdown = time()
            self.score_new = self.score - self.score_level
            self.score_level = 0

        if self.state == STATE.gameover or self.state == STATE.win:
            # Get the 5 highest scores
            self.highscores.append(self.score)
            self.highscores.sort(reverse=True)
            self.highscores = self.highscores[:5]

        if self.state == STATE.win:
            px.stop(1)  # Stop playing game music
            px.playm(4, loop=True)

    def update(self):

        t = time()
        dt = t - self.pt
        self.pt = t

        if px.btnp(px.KEY_Q):
            px.quit()

        # Uh' cheaty for debug
        # if px.btnp(px.KEY_A, 15, 15):
        #     Asteroid.asteroids.clear()
        # if px.btnp(px.KEY_Z, 15, 15):
        #     self.change_state(STATE.gameover)
        # if px.btnp(px.KEY_O, 15, 15):
        #     ovni_size = 3 - (self.level - 1) * 0.5
        #     Ovni(size=ovni_size)

        for ovni in Ovni.ovnis:
            ovni.update(dt, self.ship, self.state)

        for asteroid in Asteroid.asteroids:
            asteroid.update(dt)
        for bullet in Bullet.bullets:
            bullet.update(dt)
        for effect in self.gfx:
            if effect.over:
                self.gfx.remove(effect)
                del effect
            else:
                effect.update(dt)
        for bullet in OvniBullet.bullets:
            bullet.update(dt)

        if self.state == STATE.play:
            self.update_play(dt)
        elif self.state == STATE.title:
            self.update_title()
        elif self.state == STATE.newlevel:
            self.update_newlevel()
        elif self.state == STATE.retry:
            self.update_retry()
        elif self.state == STATE.downscore:
            self.update_downscore()
        elif self.state == STATE.gameover:
            self.update_gameover()
        elif self.state == STATE.win:
            self.update_win()

    def update_title(self):
        if px.btnp(px.KEY_RETURN):
            self.change_state(STATE.newlevel)

    def update_gameover(self):
        if px.btnp(px.KEY_RETURN):
            self.change_state(STATE.title)

    def update_win(self):

        if not len(self.gfx):
            self.ending_gfx()

        if px.btnp(px.KEY_RETURN):
            self.change_state(STATE.title)

    def update_newlevel(self):
        if self.is_countdown_ended() or px.btnp(px.KEY_RETURN, 15, 15):
            self.change_state(STATE.play)

    def update_retry(self):
        if self.is_countdown_ended():
            self.change_state(STATE.gameover)
        elif px.btnp(px.KEY_RETURN, 15, 15):
            self.change_state(STATE.downscore)

    def update_downscore(self):
        if self.score > self.score_new:
            self.score = max(self.score_new, self.score - 200)
        else:
            self.change_state(STATE.newlevel)

    def update_play(self, dt):

        self.ship.update(dt)

        if not len(Ovni.ovnis):
            if not px.frame_count % 40:  # every 1.5s 'cause framerate = 30
                # New onvi every 30s 'cause  :
                # Initial speed sound is 30 : 30-10 = 20
                # Decrease speed every 2s : 20 * 1.5 = 30
                # 0
                if px.sound(12).speed == 10:
                    # LAUNCH OVNI
                    ovni_size = 3 - (self.level - 1) * 0.5
                    Ovni(size=ovni_size)
                else:
                    px.sound(12).speed -= 1

        # HANDLE COLLISIONS
        for bullet in Bullet.bullets.copy():
            # Check if bullets hit asteroids
            for asteroid in Asteroid.asteroids.copy():
                if are_collided(bullet, asteroid):
                    bullet.hit()
                    self.asteroid_death(asteroid)
            # Check if bullets hit ovnis
            for ovni in Ovni.ovnis.copy():
                if are_collided(bullet, ovni):
                    bullet.hit()
                    self.ovni_death(ovni)

        # Check if asteroid hit ship
        for asteroid in Asteroid.asteroids.copy():
            if are_collided(asteroid, self.ship):
                self.ship_death()
                break

        # Check if ovnis hit ship
        for ovni in Ovni.ovnis.copy():
            if are_collided(ovni, self.ship) and ovni.can_fire:
                self.ship_death()
                break

        # Check if ovni bullets hit ship
        for bullet in OvniBullet.bullets:
            if are_collided(bullet, self.ship):
                bullet.hit()
                self.ship_death()
                break

        # NEW LEVEL
        # if there is nothing left then change level
        if not (len(Asteroid.asteroids)) and not (len(Ovni.ovnis)):
            self.change_state(STATE.newlevel)

    def draw(self):
        px.cls(0)

        for asteroid in Asteroid.asteroids:
            asteroid.draw()

        for bullet in Bullet.bullets:
            bullet.draw()

        for effect in self.gfx:
            effect.draw()

        for ovni in Ovni.ovnis:
            ovni.draw()
        for bullet in OvniBullet.bullets:
            bullet.draw()

        if self.state == STATE.play:
            self.ship.draw()
        elif self.state == STATE.title:
            center_msg_at_line("ASTEROID", 7, WHITE)
            center_msg_at_line("keys: right up left space", 9, GREY)
            center_msg_at_line("[ Press enter ]", 10, DARK_BLUE)

        elif self.state == STATE.newlevel:
            center_msg_at_line(f"LEVEL {self.level}", 7, GREY)
            center_msg_at_line(f"{self.level_msg[self.level]}", 8, WHITE)
            center_msg_at_line(f" Ready ?", 10, GREY)
            center_msg_at_line(f"{self.countdown}", 11, WHITE)
            center_msg_at_line("[ Enter to skip ]", 12, DARK_BLUE)

        elif self.state == STATE.retry:
            center_msg_at_line(f"GAME OVER", 7, WHITE)
            center_msg_at_line(f" Retry level {self.level} ?", 9, GREY)
            center_msg_at_line(f"{self.countdown}", 10, WHITE)
            center_msg_at_line("[ Enter to continue ]", 11, DARK_BLUE)

        elif self.state == STATE.win:
            center_msg_at_line("YOU DID IT STAR FIGHTER !", 4, WHITE)
            self.print_highscores_at_line(6)
            center_msg_at_line("[ Enter to play again ]", 13, DARK_BLUE)

        elif self.state == STATE.gameover:
            center_msg_at_line("GAME OVER", 4, WHITE)
            self.print_highscores_at_line(6)
            center_msg_at_line("[ Enter to play again ]", 13, DARK_BLUE)

        px.text(3, 3, f"Score : {self.score}", GREY)
        px.text(SCREEN_W - (80 / 2), 3, f"Level : {self.level}", GREY)


if __name__ == "__main__":
    App()
