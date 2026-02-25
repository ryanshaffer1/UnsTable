from dataclasses import dataclass, field

from pint import Quantity

from src import ureg
from src.system import Bob, DiskPrim, Pendulum


@dataclass
class TableTop(Bob, DiskPrim):
    radius: Quantity = 12 * ureg.inch
    thickness: Quantity = 2 * ureg.inch

    def __post_init__(self) -> None:
        self.moi = self.moi_through_diameter()

    def get_mass(self) -> Quantity:
        return self.mass

@dataclass
class TablePend(Pendulum):
    bob: TableTop = field(default_factory=TableTop)
