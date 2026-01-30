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
    friction_coeff: float = 0.1
    
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
    mass: Quantity = 0.5 * ureg.kg
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
            self.moi = (1/3) * self.mass * self.length**2  # Moment of inertia about pivot point
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

class Plant:
    def get_A_B(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        # Unpack variables
        b = system.b
        g = system.g
        moi = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com
        p = m_cart + m_pend - (m_pend**2 * l_com**2) / (moi + m_pend * l_com**2)  # denominator for the A and B matrices
        q = (m_pend**2 * l_com**2 - (moi + m_pend * l_com**2) * (m_cart + m_pend))/(m_pend*l_com)

        # Construct some matrix and vector elements, with units stripped for numpy
        a22 = (-b / p).magnitude
        a32 = (-g * m_pend**2 * l_com**2 / (p * (moi + m_pend * l_com**2))).magnitude
        a24 = (-b / q).magnitude
        a34 = ((-1 * (m_cart + m_pend) * g)/q).magnitude

        b2 = (1/p).magnitude
        b4 = (1/q).magnitude

        # Construct the A matrix
        A = np.array([
            [0,    1,       0,      0],
            [0,    a22,     a32,    0],
            [0,    0,       0,      1],
            [0,    a24,     a34,    0]]
            )

        # Construct the B vector
        B = np.array([
            0,
            b2,
            0,
            b4
            ])

        return A, B

    def get_A_old(self, system: System) -> np.ndarray:
        # Unpack variables
        b = system.b
        g = system.g
        moi = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com
        p = moi * (m_cart + m_pend) + m_cart * m_pend * l_com**2 # denominator for the A and B matrices

        # Construct some matrix elements, with units stripped for numpy
        a22 = (-(moi+m_pend*l_com**2)*b/p).magnitude
        a32 = ((m_pend**2*g*l_com**2)/p).magnitude
        a24 = (-(m_pend*l_com*b)/p).magnitude
        a34 = (m_pend*g*l_com*(m_cart+m_pend)/p).magnitude

        # Construct the matrix
        A = np.array([
            [0,    1,       0,      0],
            [0,    a22,     a32,    0],
            [0,    0,       0,      1],
            [0,    a24,     a34,    0]]
            )
        return A

    def get_B_old(self, system: System) -> np.ndarray:
        # Unpack variables
        moi_pend = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com
        p = moi_pend * (m_cart + m_pend) + m_cart * m_pend * l_com**2 # denominator for the A and B matrices

        # Construct some vector elements, with units stripped for numpy
        b2 = ((moi_pend + m_cart * l_com**2) / p).magnitude
        b4 = (m_cart * l_com / p).magnitude
        
        # Construct the vector
        B = np.array([
            0,
            b2,
            0,
            b4
            ])
        return B
