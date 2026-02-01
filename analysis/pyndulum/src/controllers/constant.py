from pint import Quantity

from src.controllers.abstract_controller import AbstractController
from src.primitives import State
from src.system import System

class ConstantController(AbstractController):
    def __init__(self, u: Quantity):
        self.u = u.to_base_units()
        

    def compute_u(self, system: System, state: State) -> float:
        return self.u