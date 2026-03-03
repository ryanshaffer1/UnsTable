from dataclasses import dataclass

import pandas as pd
from pint import Quantity

from src import ureg
from src.system import Actuator, Cart, Pendulum
from src.system.primitives import BodyRefPoint


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
