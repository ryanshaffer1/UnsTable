from abc import ABC, abstractmethod

from src.system import System
from src.variables import State


class AbstractController(ABC):
    @abstractmethod
    def compute_u(self, system: System, state: State) -> float:
        pass
