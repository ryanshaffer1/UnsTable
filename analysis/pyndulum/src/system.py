from dataclasses import dataclass
import numpy as np
from pint import Quantity

from src import ureg
from src.primitives import LinePrim, RectPrim, State


@dataclass
class Cart(RectPrim):
    mass: Quantity = 1 * ureg.kg
    y_top: Quantity = 0 * ureg["inch"]
    width: Quantity = 8 * ureg["inch"]
    height: Quantity = 4 * ureg["inch"]
    friction_coeff: Quantity = 0.1 * ureg.newton * ureg.second / ureg.meter
    
    def __post_init__(self):
        # Convert to base units
        self.mass.ito_base_units()
        self.y_top.ito_base_units()
        self.width.ito_base_units()
        self.height.ito_base_units()
    
    def get_ll_corner(self, state: State) -> tuple[float, float]:
        return (state.x - self.width / 2, self.y_top - self.height)

@dataclass
class Pendulum(LinePrim):
    mass: Quantity = 2 * ureg.kg
    length: Quantity = 24 * ureg["inch"]
    y_pivot: Quantity = 0 * ureg["inch"]
    thickness: Quantity = 3 * ureg["inch"]
    moi: Quantity | None = None  # Moment of inertia about pivot point
    length_pivot_to_centroid: Quantity | None = None  # Distance from pivot to center of mass
    
    def __post_init__(self):
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

    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        cart_x = state.x
        theta = state.theta
        bob_x = cart_x + self.length * np.sin(theta)
        bob_y = self.y_pivot + self.length * np.cos(theta)
        return (cart_x, self.y_pivot), (bob_x, bob_y)

@dataclass
class System:
    cart: Cart
    pendulum: Pendulum
    gravity: Quantity = 9.81 * ureg.meter / ureg.second**2

    def __post_init__(self):
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


    def trace_pend_endpoint_history(self, state_history: np.ndarray) -> np.ndarray:
        # Allocate array for output
        endpoint_history = np.zeros([4, state_history.shape[1]])
        # Iterate through states in state history
        for i, row in enumerate(state_history.T):
            # Calculate and store the tip position
            base_pos, tip_pos = self.pendulum.get_endpoints(State.from_vector(row))
            endpoint_history[0:2, i] = Quantity.from_sequence(base_pos).to_base_units().magnitude
            endpoint_history[2:4, i] = Quantity.from_sequence(tip_pos).to_base_units().magnitude
        return endpoint_history