from dataclasses import dataclass, field

import numpy as np
from pint import Quantity

from src import ureg
from src.system import BeamPrim, BlockPrim, ObjectPrim
from src.variables import State


@dataclass
class Actuator:
    force_limit: Quantity | None = None
    refresh_rate: Quantity | None = None
    command_lag: Quantity | None = None


    def enforce_limit(self, u: Quantity) -> Quantity:
        if self.force_limit and abs(u) > self.force_limit:
            u = np.sign(u) * self.force_limit

        return u

    def is_update_time(self, time: Quantity, dt: Quantity) -> bool:
        return (self.refresh_rate is None) or (dt==0) or (time % (1/self.refresh_rate) < dt)

    def is_past_lag_time(self, time: Quantity, lag_window_start: Quantity) -> bool:
        return (self.command_lag is None) or (time - lag_window_start >= self.command_lag)

@dataclass
class Cart(BlockPrim):
    mass: Quantity = 1 * ureg.kg
    y_top: Quantity = 0 * ureg["inch"]
    width: Quantity = 8 * ureg["inch"]
    height: Quantity = 4 * ureg["inch"]
    friction_coeff: Quantity = 0.1 * ureg.newton * ureg.second / ureg.meter

    def __post_init__(self) -> None:
        # Convert to base units
        self.mass.ito_base_units()
        self.y_top.ito_base_units()
        self.width.ito_base_units()
        self.height.ito_base_units()

    def get_mass(self) -> Quantity:
        return self.mass

    def get_ll_corner(self, state: State) -> tuple[float, float]:
        return (state.x - self.width / 2, self.y_top - self.height)

@dataclass
class Bob(ObjectPrim):
    mass: Quantity = 0 * ureg.kg  # Mass of pendulum bob (if separate from rod mass)
    moi: Quantity = 0 * ureg.kg * ureg.meter**2  # Pendulum bob MOI about its center of mass
    cg_tip_offset_x: Quantity = 0 * ureg.meter  # Distance from pendulum tip to bob center of mass
    cg_tip_offset_y: Quantity = 0 * ureg.meter  # Distance from pendulum tip to bob center of mass

    def moi_through_pivot(self, rod_length: Quantity) -> Quantity:
        dist_pivot_to_cg = np.sqrt(self.cg_tip_offset_x**2 + (rod_length + self.cg_tip_offset_y)**2)
        return self.moi + self.parallel_axis_term(dist_pivot_to_cg)

@dataclass
class Rod(BeamPrim):
    mass: Quantity = 2 * ureg.kg
    length: Quantity = 24 * ureg.inch
    thickness: Quantity = 3 * ureg.inch

    def __post_init__(self) -> None:
        # Convert to base units
        self.mass.ito_base_units()
        self.length.ito_base_units()
        self.thickness.ito_base_units()

    def get_mass(self) -> Quantity:
        return self.mass

    @ureg.wraps(("meter", "meter"), (None, "meter", "radian"), strict=False)
    def get_endpoints(self, x: float, theta: float,
                      ) -> tuple[tuple[float, float], tuple[float, float]]:
        bob_x = x + self.length.magnitude * np.sin(theta)
        bob_y = self.length.magnitude * np.cos(theta)
        return (x, 0), (bob_x, bob_y)


@dataclass
class Pendulum:
    y_pivot: Quantity = 0 * ureg["inch"] # Height of pendulum pivot point (relative to ground)
    moi: Quantity | None = None  # Moment of inertia about pivot point
    centroid: tuple[Quantity, Quantity] | None = None  # Distance from pivot to center of mass
    rod: Rod = field(default_factory=Rod)
    bob: Bob = field(default_factory=Bob)

    def __post_init__(self) -> None:
        self.mass = self.rod.mass + self.bob.mass

        # Convert to base units
        self.y_pivot.ito_base_units()

        # Calculate distance from pivot to center of mass if not provided
        if self.centroid is None:
            self.centroid = self.get_centroid_offsets()

        # Calculate moment of inertia about pivot point if not provided
        if self.moi is None:
            self.moi = self.get_moi_about_pivot()

    def get_endpoints(self,
                      x: Quantity,
                      theta: Quantity,
                      ) -> tuple[tuple[Quantity, Quantity], tuple[Quantity, Quantity]]:
        # Get rod endpoints
        rod_start, rod_end = self.rod.get_endpoints(x, theta)
        # Add y_pivot offset to both endpoints
        rod_start = (rod_start[0], rod_start[1] + self.y_pivot)
        rod_end = (rod_end[0], rod_end[1] + self.y_pivot)

        return rod_start, rod_end

    def get_centroid_offsets(self) -> tuple[Quantity, Quantity]:
        bob_centroid = [self.bob.cg_tip_offset_x, self.rod.length + self.bob.cg_tip_offset_y]
        rod_centroid = self.rod.get_centroid_offset()

        offset_x = (self.bob.mass * bob_centroid[0] + self.rod.mass * rod_centroid[0]) / self.mass
        offset_y = (self.bob.mass * bob_centroid[1] + self.rod.mass * rod_centroid[1]) / self.mass
        return (offset_x, offset_y)

    def get_moi_about_pivot(self) -> Quantity:
        # Total moment of inertia
        return self.rod.moi_through_endpoint() + self.bob.moi_through_pivot(self.rod.length)
