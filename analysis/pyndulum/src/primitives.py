from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

import numpy as np
from pint import Quantity

from src import ureg


@dataclass
class State:
    x: Quantity
    vx: Quantity
    theta: Quantity
    omega: Quantity

    def to_vector(self) -> list[Quantity]:
        return np.array([self.x.to_base_units().magnitude,
                         self.vx.to_base_units().magnitude,
                         self.theta.to_base_units().magnitude,
                         self.omega.to_base_units().magnitude])

    @classmethod
    def from_vector(cls, vector: list[float]) -> Self:
        return cls(x = vector[0] * ureg.meter,
                   vx = vector[1] * ureg.meter / ureg.second,
                   theta = vector[2] * ureg.radian,
                   omega = vector[3] * ureg.radian / ureg.second)

    def to_display_units(self) -> dict[str, Quantity]:
        return {
            "x": self.x.to(ureg.meter),
            "vx": self.vx.to(ureg.meter / ureg.second),
            "theta": self.theta.to(ureg.degree),
            "omega": self.omega.to(ureg.degree / ureg.second),
        }

    def add_vector(self, vector: np.ndarray) -> Self:
        x, vx, theta, omega = self.add_vector_unitless(
            self.x, self.vx, self.theta, self.omega,
            vector[0], vector[1], vector[2], vector[3],
        )

        return State(x, vx, theta, omega)

    @ureg.wraps((ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ),
                (None,
                 ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ureg.meter,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ),
                strict=False)
    def add_vector_unitless(self, # noqa: PLR0913
                            x1: float,
                            vx1: float,
                            theta1: float,
                            omega1: float,
                            x2: float,
                            vx2: float,
                            theta2: float,
                            omega2: float,
                            ) -> tuple[float, float, float, float]:
        x = x1 + x2
        vx = vx1 + vx2
        theta = theta1 + theta2
        omega = omega1 + omega2
        return x, vx, theta, omega

class RectPrim(ABC):
    width: Quantity
    height: Quantity

    @abstractmethod
    def get_ll_corner(self, state: State) -> tuple[float, float]:
        pass

    def get_width(self) -> Quantity:
        return self.width

    def get_height(self) -> Quantity:
        return self.height

class LinePrim(ABC):
    length: Quantity
    thickness: Quantity

    @abstractmethod
    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        pass
