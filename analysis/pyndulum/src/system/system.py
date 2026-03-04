from dataclasses import dataclass

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
        self.l_com = self.pendulum.centroid.z


    def get_bounding_box(self, history: pd.DataFrame) -> tuple[np.ndarray]:
        # Initialize min/max values for X, Y, and Z limits
        bounding_box = np.array(((np.nan, -np.nan),
                                 (np.nan, -np.nan),
                                 (np.nan, -np.nan)))*ureg.meter

        # Iterate through all states
        for _, row in history.iterrows():
            state = State(*row[["x","vx","theta","omega"]].to_numpy())
            # Get bounding boxes for the system components given the current state
            bbox_cart = self.cart.global_bounding_box(state)
            bbox_pend = self.pendulum.global_bounding_box(state)
            # Update the overall bounding box
            bounding_box = update_bounding_box(bounding_box,
                                               np.concatenate((bbox_cart, bbox_pend), axis=1))

        return tuple(bounding_box)
