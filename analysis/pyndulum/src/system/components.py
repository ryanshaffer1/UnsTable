from dataclasses import dataclass, field

import numpy as np
from pint import Quantity

from src import ureg
from src.coords import CoordFrame, Point
from src.system.rigid_bodies import (
    Block,
    BodyRefPoint,
    Cylinder,
    RigidBody,
    RigidBodySystem,
    Sphere,
    )


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
class Cart(Block):
    mass: Quantity = 1 * ureg.kg
    width: Quantity = 8 * ureg["inch"]
    height: Quantity = 4 * ureg["inch"]
    depth: Quantity = 8 * ureg["inch"]
    origin_type: BodyRefPoint = BodyRefPoint.TOP_CENTER
    body_frame: CoordFrame = field(default_factory=CoordFrame)
    friction_coeff: Quantity = 0.1 * ureg.newton * ureg.second / ureg.meter

    def __post_init__(self) -> None:
        super().__post_init__()

@dataclass
class Bob(Sphere):
    """Convenience class setting default values for a spherical bob."""
    mass: Quantity = 0 * ureg.kg  # Mass of pendulum bob (if separate from rod mass)
    radius: Quantity = 4 * ureg.inches
    body_frame: CoordFrame = field(default_factory=CoordFrame)
    origin_type: BodyRefPoint = BodyRefPoint.BOTTOM_CENTER

@dataclass
class Rod(Cylinder):
    """Convenience class setting default values for a cylinder."""
    mass: Quantity = 2 * ureg.kg
    length: Quantity = 24 * ureg.inch
    radius: Quantity = 1 * ureg.inch
    origin_type: BodyRefPoint = BodyRefPoint.BOTTOM_CENTER
    body_frame: CoordFrame = field(default_factory=CoordFrame)

@dataclass
class Pendulum(RigidBodySystem):
    rod: Rod = field(default_factory=Rod)
    bob: Bob | None = None
    bodies: list[RigidBody] = field(init=False)
    body_frame: CoordFrame = field(default_factory=CoordFrame)
    origin_type: None = None

    def __post_init__(self) -> None:
        # Control rotation axes
        self.rotations = {"Y": "theta"}

        # Set up sub-bodies and position their body frames
        self.bodies = [self.rod]
        if self.bob is not None:
            self.bodies.append(self.bob)
            if self.bob.origin_mount is None:
                self.bob.origin_mount = [self.rod, BodyRefPoint.TOP_CENTER]
                self.bob.set_origin_from_other_body(*self.bob.origin_mount)
        super().__post_init__()

    def set_pivot_point(self, pivot_point: Point) -> None:
        self.body_frame.translate_to(point=pivot_point.to_global())
        self.moi = self.get_moi_matrix(self.body_frame.origin)
