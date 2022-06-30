""" 
The MIT License (MIT)

Copyright (c) 2022 Antony Templier

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from math import sqrt, atan2, cos, sin, acos, pi
from random import random
from typing import Union, List


class Vector2D:

    ZERO: "Vector2D"
    UP: "Vector2D"
    RIGHT: "Vector2D"
    DOWN: "Vector2D"
    LEFT: "Vector2D"

    def __init__(self, x: Union[int, float] = 0, y: Union[int, float] = 0) -> None:
        self.x = x
        self.y = y

    @classmethod
    def from_angle(cls, theta):
        """Returns a new unit 2D vector from the specified angle value
        theta is an angle in radian
        theta is clockwise and relative to positive Y
        """
        return cls(sin(theta), cos(theta))

    @classmethod
    def random(cls):
        """Return a random unit vector"""
        return cls.from_angle(random() * pi * 2)

    @property
    def values(self) -> tuple:
        return (self.x, self.y)

    @property
    def norm_square(self) -> Union[int, float]:
        return self.x * self.x + self.y * self.y

    @property
    def norm(self) -> Union[int, float]:
        return sqrt(self.x * self.x + self.y * self.y)

    @norm.setter
    def norm(self, magnitude: Union[int, float]):
        self.normalize()
        self.mul(magnitude)

    @property
    def argument(self) -> float:
        """return angle in radians clockwise from positive y axe"""
        return atan2(self.x, self.y)

    @argument.setter
    def argument(self, theta: Union[int, float]):
        """Set argument of vector
        theta is an angle in radian
        theta is clockwise and relative to positive Y"""
        n = self.norm
        self.x = -sin(theta) * n
        self.y = cos(theta) * n

    def copy(self):
        return Vector2D(self.x, self.y)

    def __len__(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f"<Vector2D{self.values} at {hex(id(self))}>"

    def __str__(self) -> str:
        return str(self.values)

    def __getitem__(self, key: int):
        if key == 0:
            return self.x
        elif key == 1:
            return self.y
        raise IndexError("Vector2D arguments out of range")

    def abs(self) -> None:
        self.x = abs(self.x)
        self.y = abs(self.y)

    def normalize(self) -> None:
        """Transform to a unit vector"""
        norm = self.norm
        self.x /= norm
        self.y /= norm

    def reverse(self) -> None:
        """Transform to the opposite vector"""
        self.x = -self.x
        self.y = -self.y

    def rotate(self, theta: Union[int, float]) -> None:
        """Rotate with a given angle in radian from the positive y axe"""
        tcos, tsin = cos(theta), sin(theta)
        x, y = self.x, self.y
        self.x = tcos * x - tsin * y
        self.y = tsin * x + tcos * y

    def add(self, scalar: Union[int, float]) -> None:
        """Add a scalar to vector components"""
        self.x += scalar
        self.y += scalar

    def vadd(self, other: Union["Vector2D", tuple, List]) -> None:
        """Add a list, a tuple or another Vector2D to vector components"""
        self.x += other[0]
        self.y += other[1]

    def sub(self, scalar: Union[int, float]) -> None:
        """Substract a scalar to vector components"""
        self.x -= scalar
        self.y -= scalar

    def vsub(self, other: Union["Vector2D", tuple, List]) -> None:
        """Substract a list, a tuple or another Vector2D to vector components"""
        self.x -= other[0]
        self.y -= other[1]

    def mul(self, scalar: Union[int, float]) -> None:
        """Multiply each vector components with a scalar"""
        self.x *= scalar
        self.y *= scalar

    def vmul(self, other: Union["Vector2D", tuple, List]) -> None:
        """Multiply each vector components with each other's members"""
        self.x *= other[0]
        self.y *= other[1]

    def div(self, scalar: Union[int, float]) -> None:
        """Divide each vector components with a scalar"""
        self.x /= scalar
        self.y /= scalar

    def vdiv(self, other: Union["Vector2D", tuple, List]) -> None:
        """Divide each vector components with each other's members"""
        self.x /= other[0]
        self.y /= other[1]

    def mod(self, scalar: Union[int, float]) -> None:
        """Divide each vector components with a scalar"""
        self.x %= scalar
        self.y %= scalar

    def vmod(self, other: Union["Vector2D", tuple, List]) -> None:
        """Divide each vector components with each other's members"""
        self.x %= other[0]
        self.y %= other[1]

    def dot(self, other: "Vector2D") -> Union[int, float]:
        """Return the dot prodct of two vectors"""
        return self.x * other.x + self.y * other.y

    def dist_between(self, other: "Vector2D") -> Union[int, float]:
        """Distance between two vectors"""
        v = Vector2D(other.x - self.x, other.y - self.y)
        return v.norm

    def angle_between(self, other: "Vector2D") -> float:
        """Angle in radians between two vectors"""
        v1 = Vector2D(self.x, self.y)
        v2 = Vector2D(other.x, other.y)
        v1.normalize()
        v2.normalize()
        return acos(v1.dot(v2))

    def to_polar(self) -> "Vector2D":
        """Return tuple containing (norm, argument)"""
        return Vector2D(self.norm, self.argument)

    def clamp(
        self,
        minvec: Union["Vector2D", tuple, List],
        maxvec: Union["Vector2D", tuple, List],
    ) -> None:
        """Restrict to a given range"""
        self.x = min(max(minvec[0], self.x), maxvec[0])
        self.y = min(max(minvec[1], self.y), maxvec[1])

    def limit(self, maximum: Union[int, float]):
        """Restric the norm to that limit"""
        if self.norm_square > maximum * maximum:
            self.normalize()
            self.mul(maximum)

    def lerp(self, other: "Vector2D", amount) -> "Vector2D":
        return self + (other - self) * amount

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Addition with Vectors
        Returns a new vector"""
        return Vector2D(self.x + other.x, self.y + other.y)

    def __iadd__(self, other: "Vector2D") -> "Vector2D":
        """Addition with Vectors
        Returns the instance"""
        self.x += other.x
        self.y += other.y
        return self

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Substraction with Vectors
        Returns a new vector"""
        return Vector2D(self.x - other.x, self.y - other.y)

    def __isub__(self, other: "Vector2D") -> "Vector2D":
        """Substraction with Vectors
        Returns the instance"""
        self.x -= other.x
        self.y -= other.y
        return self

    def __mul__(self, other: Union[int, float]) -> "Vector2D":
        """Multiplication with a scalar. Returns always a new vector"""
        return Vector2D(self.x * other, self.y * other)

    def __imul__(self, other: Union[int, float]) -> "Vector2D":
        """Multiplication with a scalar. Returns always the instance"""
        self.x *= other
        self.y *= other
        return self

    def __truediv__(self, other: Union[int, float]) -> "Vector2D":
        """Division with a scalar. Returns always a new vector"""
        return Vector2D(self.x / other, self.y / other)

    def __itruediv__(self, other: Union[int, float]) -> "Vector2D":
        """Division with a scalar. Returns always the instance"""
        self.x /= other
        self.y /= other
        return self

    def __mod__(self, other: Union[int, float]) -> "Vector2D":
        """Division with a scalar. Returns always a new vector"""
        return Vector2D(self.x % other, self.y % other)

    def __imod__(self, other: Union[int, float]) -> "Vector2D":
        """Division with a scalar. Returns always the instance"""
        self.x %= other
        self.y %= other
        return self

    __matmul__ = dot

    def __neg__(self) -> "Vector2D":
        return Vector2D(-self.x, -self.y)

    def __eq__(self, other) -> bool:
        if type(other) in [Vector2D, tuple, list]:
            return self.x == other[0] and self.y == other[1]
        return False

    def __abs__(self) -> "Vector2D":
        return Vector2D(abs(self.x), abs(self.y))

    def __hash__(self) -> int:
        return hash((self.x, self.y))


Vector2D.ZERO = Vector2D(0, 0)
Vector2D.UP = Vector2D(0, -1)
Vector2D.RIGHT = Vector2D(1, 0)
Vector2D.DOWN = Vector2D(0, 1)
Vector2D.LEFT = Vector2D(-1, 0)
