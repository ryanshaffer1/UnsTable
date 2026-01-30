from abc import ABC, abstractmethod
from dataclasses import dataclass

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
        
    def to_display_units(self) -> dict[str, Quantity]:
        return {
            "x": self.x.to(ureg.meter),
            "vx": self.vx.to(ureg.meter / ureg.second),
            "theta": self.theta.to(ureg.degree),
            "omega": self.omega.to(ureg.degree / ureg.second)
        }

class Process(ABC):
    state: State

    @abstractmethod
    def update(self, time: float) -> None:
        pass

    def get_state(self) -> State:
        return self.state

class RectPrim(ABC):
    width: Quantity
    height: Quantity

    @abstractmethod
    def get_ll_corner(self, state: State) -> tuple[float, float]:
        pass
        
    def get_width(self):
        return self.width

    def get_height(self):
        return self.height

class LinePrim(ABC):
    length: Quantity
    thickness: Quantity

    @abstractmethod
    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        pass