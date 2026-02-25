from dataclasses import dataclass

import pandas as pd
from pint import Quantity

from src import ureg
from src.system import Actuator, Cart, Pendulum


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
        self.l_com = self.pendulum.centroid[1]


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
