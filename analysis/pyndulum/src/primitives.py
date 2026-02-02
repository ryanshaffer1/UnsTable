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
