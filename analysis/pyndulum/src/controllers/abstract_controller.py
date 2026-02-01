from abc import ABC, abstractmethod

import numpy as np
from pint import Quantity

from src.primitives import State
from src.system import System

class AbstractController(ABC):
    def __init__(self, limit: Quantity | None = None) -> None:
        self.limit = limit
    
    @abstractmethod
    def compute_u(self, system: System, state: State) -> float:
        pass


    def enforce_limit(self, u: Quantity) -> Quantity:
        if self.limit and abs(u) > self.limit:
            u = np.sign(u) * self.limit
        
        return u