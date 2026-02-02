from pint import Quantity

from src.controllers.abstract_controller import AbstractController
from src.system import System


class ConstantController(AbstractController):
    def __init__(self, u: Quantity) -> None:
        self.u = u.to_base_units()


    def compute_u(self, system: System, *args: tuple, **kwargs: dict) -> float:
        return system.actuator.enforce_limit(self.u)
