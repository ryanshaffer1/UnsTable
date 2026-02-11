from dataclasses import dataclass

import numpy as np
import pandas as pd
from pint import Quantity

from src import ureg
from src.primitives import BeamPrim, BlockPrim, State


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

    def get_ll_corner(self, state: State) -> tuple[float, float]:
        return (state.x - self.width / 2, self.y_top - self.height)

@dataclass
class Pendulum(BeamPrim):
    mass: Quantity = 2 * ureg.kg
    length: Quantity = 24 * ureg["inch"]
    y_pivot: Quantity = 0 * ureg["inch"]
    thickness: Quantity = 3 * ureg["inch"]
    moi: Quantity | None = None  # Moment of inertia about pivot point
    length_pivot_to_centroid: Quantity | None = None  # Distance from pivot to center of mass

    def __post_init__(self) -> None:
        # Convert to base units
        self.mass.ito_base_units()
        self.length.ito_base_units()
        self.y_pivot.ito_base_units()
        self.thickness.ito_base_units()

        # Calculate moment of inertia and length to centroid if not provided
        if self.moi is None:
            self.moi = (1/12) * self.mass * self.length**2  # Moment of inertia about pivot point
        if self.length_pivot_to_centroid is None:
            self.length_pivot_to_centroid = self.length / 2  # Distance from pivot to center of mass

    @ureg.wraps(("meter", "meter"), (None, "meter", "radian"), strict=False)
    def get_endpoints(self, x: float, theta: float,
                      ) -> tuple[tuple[float, float], tuple[float, float]]:
        bob_x = x + self.length.magnitude * np.sin(theta)
        bob_y = self.y_pivot.magnitude + self.length.magnitude * np.cos(theta)
        return (x, self.y_pivot.magnitude), (bob_x, bob_y)

@dataclass
class System:
    actuator: Actuator
    cart: Cart
    pendulum: Pendulum
    gravity: Quantity = 9.81 * ureg.meter / ureg.second**2

    def __post_init__(self) -> None:
        # Totals
        self.total_mass = self.cart.mass + self.pendulum.mass

        # Configure geometry parameters dependent on other components
        self.pendulum.y_pivot = self.cart.y_top

        # Set variables used in dynamics calculations
        self.b = self.cart.friction_coeff
        self.g = self.gravity
        self.moi_pend = self.pendulum.moi
        self.m_cart = self.cart.mass
        self.m_pend = self.pendulum.mass
        self.l_com = self.pendulum.length_pivot_to_centroid


    def trace_pend_endpoint_history(self, history: pd.DataFrame) -> pd.DataFrame:
        # Calculate the pendulum endpoint history based on the state history and system parameters
        endpoint_history = history[["x","theta"]].apply(lambda row:
            self.pendulum.get_endpoints(row["x"], row["theta"]),
            axis=1, result_type="expand")
        endpoint_history = endpoint_history.assign(base_x=endpoint_history[0].apply(lambda p: p[0]),
                                                   base_y=endpoint_history[0].apply(lambda p: p[1]),
                                                   tip_x=endpoint_history[1].apply(lambda p: p[0]),
                                                   tip_y=endpoint_history[1].apply(lambda p: p[1]))
        return endpoint_history
