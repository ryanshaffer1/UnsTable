from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections, patches
from pint import Quantity

from src import ureg
from src.coords import CoordSystem, GlobalPoint, Point
from src.system.primitives import (
    Block,
    BodyRefPoint,
    Cylinder,
    RigidBody,
    RigidBodySystem,
    Sphere,
    )
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
class Cart(Block):
    mass: Quantity = 1 * ureg.kg
    width: Quantity = 8 * ureg["inch"]
    height: Quantity = 4 * ureg["inch"]
    depth: Quantity = 8 * ureg["inch"]
    origin_type: BodyRefPoint = BodyRefPoint.TOP_CENTER
    body_frame: CoordSystem = field(default_factory=CoordSystem)
    friction_coeff: Quantity = 0.1 * ureg.newton * ureg.second / ureg.meter

    def __post_init__(self) -> None:
        super().__post_init__()

    def get_mpl_sprite(self, **kwargs: dict) -> patches.Rectangle:
        return patches.Rectangle((0,0), self.width, self.height, **kwargs)

@dataclass
class Bob(Sphere):
    mass: Quantity = 0 * ureg.kg  # Mass of pendulum bob (if separate from rod mass)
    radius: Quantity = 2 * ureg.inches
    body_frame: CoordSystem = field(default_factory=CoordSystem)
    origin_type: BodyRefPoint = BodyRefPoint.BOTTOM_CENTER

    def __post_init__(self) -> None:
        super().__post_init__()
        self.body_frame.set_rotations((("Y", "theta"),)) # TODO make this happen in the Pendulum

    def get_mpl_sprite(self, **kwargs: dict) -> patches.Circle:
        return patches.Circle((0,0), self.radius, **kwargs)

@dataclass
class Rod(Cylinder):
    mass: Quantity = 2 * ureg.kg
    length: Quantity = 24 * ureg.inch
    radius: Quantity = 1 * ureg.inch
    origin_type: BodyRefPoint = BodyRefPoint.BOTTOM_CENTER
    body_frame: CoordSystem = field(default_factory=CoordSystem)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.body_frame.set_rotations((("Y", "theta"),)) # TODO make this happen in the Pendulum

    @ureg.wraps(("meter", "meter"), (None, "meter", "radian"), strict=False)
    def get_endpoints(self, x: float, theta: float,
                      ) -> tuple[tuple[float, float], tuple[float, float]]:
        # TODO: rewrite using primitives
        bob_x = x + self.length.magnitude * np.sin(theta)
        bob_y = self.length.magnitude * np.cos(theta)
        return (x, 0), (bob_x, bob_y)

    def get_mpl_sprite(self, **kwargs: dict) -> patches.Rectangle:
        return patches.Rectangle((0,0), 2*self.radius, self.length, **kwargs)

@dataclass
class Pendulum(RigidBodySystem):
    pivot_point: Point | None = None
    rod: Rod = field(default_factory=Rod)
    bob: Bob = field(default_factory=Bob)
    # bob: Bob | None = None
    bodies: list[RigidBody] = field(init=False)
    body_frame: CoordSystem = field(default_factory=CoordSystem)
    origin_type: None = None

    def __post_init__(self) -> None:
        # Set up sub-bodies and link their body frames
        self.bodies = [self.rod]
        if self.bob is not None:
            self.bodies.append(self.bob)
            self.bob.body_frame.set_origin_point(self.rod.get_point(BodyRefPoint.TOP_CENTER,
                                                                    cs_type="body"))
        for body in self.bodies:
            body.parent_frame = self.body_frame

        # Get pivot point from CS origin
        if self.pivot_point is None:
            self.pivot_point = self.body_frame.get_framed_point()

        # Caclulate mass properties
        self.mass = self.get_mass()
        self.centroid = self.get_centroid()
        self.moi = self.get_moi_matrix(self.pivot_point)

    def set_pivot_point(self, pivot_point: Point) -> None:
        self.pivot_point = pivot_point
        self.moi = self.get_moi_matrix(self.pivot_point)

    def get_endpoints(self,
                      x: Quantity,
                      theta: Quantity,
                      ) -> tuple[tuple[Quantity, Quantity], tuple[Quantity, Quantity]]:
        # Get rod endpoints
        rod_start, rod_end = self.rod.get_endpoints(x, theta)
        # Add pivot point offset to both endpoints
        rod_start = (rod_start[0], rod_start[1] + self.pivot_point.y)
        rod_end = (rod_end[0], rod_end[1] + self.pivot_point.y)

        return rod_start, rod_end

    def get_mpl_sprite(self, **kwargs: dict) -> list[patches.Patch]:
        patches = [body.get_mpl_sprite(**kwargs) for body in self.bodies]
        return patches
