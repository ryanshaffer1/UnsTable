from abc import ABC, abstractmethod

from src.primitives import State
from src.sim_components import System

class AbstractController(ABC):
    @abstractmethod
    def compute_u(self, system: System, state: State) -> float:
        pass

