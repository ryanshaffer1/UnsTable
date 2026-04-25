from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from pint import Quantity

from src import ureg
from src.system import Actuator, Cart, Pendulum
from src.system.rigid_bodies import BodyRefPoint, update_bounding_box
from src.variables import State


@dataclass
class System:
    actuator: Actuator
    cart: Cart
    pendulum: Pendulum
    gravity: Quantity = 9.81 * ureg.meter / ureg.second**2
    times: tuple[Quantity, Quantity] | None = None

    def __post_init__(self) -> None:
        # Totals
        self.total_mass = self.cart.mass + self.pendulum.get_mass()

        # Configure geometry parameters dependent on other components
        pivot_point = self.cart.get_point(BodyRefPoint.TOP_CENTER, cs_type="body")
        self.pendulum.set_pivot_point(pivot_point)

        # Set variables used in dynamics calculations
        self.b = self.cart.friction_coeff
        self.g = self.gravity
        self.moi_pend = self.pendulum.moi[1][1] # Y-axis MOI
        self.m_cart = self.cart.mass
        self.m_pend = self.pendulum.mass
        self.l_com = np.linalg.norm(self.pendulum.centroid.vector_to(pivot_point))

    def update_state_during_transition(self, other: Self, state: State) -> State:
        # Update the state theta so that the pendulum "body" is in the same place
        # from one system to the next (theta passes through the centroid, which is changing)
        self_theta_offset = self.pendulum.rotation_offsets.get("Y",0*ureg.radian)
        other_theta_offset = other.pendulum.rotation_offsets.get("Y",0*ureg.radian)
        state.theta += (self_theta_offset - other_theta_offset)
        return state

    def valid_time(self, time: Quantity) -> bool:
        if self.times is None:
            return True
        return self.times[0] <= time < self.times[1]

    def get_bounding_box(self, history: pd.DataFrame) -> tuple[np.ndarray]:
        # Initialize min/max values for X, Y, and Z limits
        bounding_box = np.array(((np.nan, -np.nan),
                                 (np.nan, -np.nan),
                                 (np.nan, -np.nan)))*ureg.meter

        # Iterate through all states
        for time, row in history.iterrows():
            if not self.valid_time(time):
                continue
            state = State(*row[["x","vx","theta","omega"]].to_numpy())
            # Get bounding boxes for the system components given the current state
            bbox_cart = self.cart.global_bounding_box(state)
            bbox_pend = self.pendulum.global_bounding_box(state)
            # Update the overall bounding box
            bounding_box = update_bounding_box(bounding_box,
                                               np.concatenate((bbox_cart, bbox_pend), axis=1))

        return tuple(bounding_box)
