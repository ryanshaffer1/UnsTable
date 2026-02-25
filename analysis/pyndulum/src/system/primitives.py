
from abc import ABC, abstractmethod

from pint import Quantity

from src import ureg
from src.variables import State


class ObjectPrim(ABC):
    @abstractmethod
    def get_mass(self) -> Quantity:
        pass

    @abstractmethod
    def get_moi(self, direction: str, offset_dist: Quantity) -> Quantity:
        pass

    def parallel_axis_term(self, offset_dist: Quantity | None) -> Quantity:
        if offset_dist is None:
            return 0 * ureg.kg * ureg.meter**2
        return self.get_mass() * offset_dist**2

class BlockPrim(ObjectPrim, ABC):
    width: Quantity
    height: Quantity

    @abstractmethod
    def get_ll_corner(self, state: State) -> tuple[float, float]:
        pass

    def get_width(self) -> Quantity:
        return self.width

    def get_height(self) -> Quantity:
        return self.height

    def get_moi(self, direction: str, offset_dist: Quantity | None = None) -> Quantity:
        if direction == "center":
            return self.moi_through_center() + self.parallel_axis_term(offset_dist)
        if direction == "base":
            return self.moi_through_base() + self.parallel_axis_term(offset_dist)
        msg = f"Invalid direction '{direction}' for MOI calculation"
        raise ValueError(msg)

    def moi_through_center(self) -> Quantity:
        return (1/12) * self.get_mass() * (self.width**2 + self.height**2)

    def moi_through_base(self) -> Quantity:
        return self.moi_through_center() + self.parallel_axis_term(self.height / 2)

class BeamPrim(ObjectPrim, ABC):
    length: Quantity
    thickness: Quantity

    @abstractmethod
    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        pass

    @abstractmethod
    def get_mass(self) -> Quantity:
        pass

    def get_moi(self, direction: str, offset_dist: Quantity | None = None) -> Quantity:
        if direction == "center":
            return self.moi_through_center() + self.parallel_axis_term(offset_dist)
        if direction == "endpoint":
            return self.moi_through_endpoint() + self.parallel_axis_term(offset_dist)
        msg = f"Invalid direction '{direction}' for MOI calculation"
        raise ValueError(msg)

    def moi_through_center(self) -> Quantity:
        return (1/12) * self.get_mass() * self.length**2

    def moi_through_endpoint(self) -> Quantity:
        return self.moi_through_center() + self.parallel_axis_term(self.length / 2)

    def get_centroid_offset(self) -> tuple[Quantity, Quantity]:
        return 0*ureg.meter, self.length / 2

class DiskPrim(ObjectPrim, ABC):
    radius: Quantity
    thickness: Quantity

    @abstractmethod
    def get_mass(self) -> Quantity:
        pass

    def get_moi(self, direction: str, offset_dist: Quantity | None = None) -> Quantity:
        if direction == "center":
            return self.moi_through_center() + self.parallel_axis_term(offset_dist)
        if direction == "diameter":
            return self.moi_through_diameter() + self.parallel_axis_term(offset_dist)
        msg = f"Invalid direction '{direction}' for MOI calculation"
        raise ValueError(msg)

    def moi_through_center(self) -> Quantity:
        return 0.5 * self.radius**2 * self.get_mass()

    def moi_through_diameter(self) -> Quantity:
        return 0.25 * self.radius**2 * self.get_mass()

    def get_centroid_offset(self) -> tuple[Quantity, Quantity]:
        return 0*ureg.meter, self.thickness / 2
