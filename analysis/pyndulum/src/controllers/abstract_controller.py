from abc import ABC, abstractmethod

from src.primitives import State
from src.system import System

class AbstractController(ABC):
    @abstractmethod
    def compute_u(self, system: System, state: State) -> float:
        pass

