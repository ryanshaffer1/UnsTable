from abc import ABC, abstractmethod

import numpy as np
from pint import Quantity

from src.primitives import State
from src.system import System


class AbstractDynamicsModel(ABC):
    @abstractmethod
    def calc_state_derivative(self, state: State, system: System, u: Quantity) -> np.ndarray:
        pass

    def state_derivative_to_vector(self,
                                   x_dot: Quantity,
                                   x_ddot: Quantity,
                                   theta_dot: Quantity,
                                   theta_ddot: Quantity,
                                   ) -> np.ndarray:
        return np.array([x_dot.magnitude,
                         x_ddot.magnitude,
                         theta_dot.magnitude,
                         theta_ddot.magnitude])


class LinearizedModel(AbstractDynamicsModel):
    def __init__(self) -> None:
        pass

    def get_A_B(self, system: System) -> tuple[np.ndarray, np.ndarray]:
        # Unpack variables
        b = system.b
        g = system.g
        moi = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com
        # Denominators for the A and B matrix elements
        p = m_cart + m_pend - (m_pend**2 * l_com**2) / (moi + m_pend * l_com**2)
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
            [0,    a24,     a34,    0]],
            )

        # Construct the B vector
        B = np.array([
            0,
            b2,
            0,
            b4,
            ]).reshape(-1,1)

        return A, B

    def calc_state_derivative(self, state: State, system: System, u: Quantity) -> np.ndarray:
        # Construct A and B matrices from plant
        A, B = self.get_A_B(system)

        # Compute state derivative
        state_deriv = A @ state.to_vector() + B * u.magnitude
        return state_deriv


class NonlinearModel(AbstractDynamicsModel):
    def __init__(self) -> None:
        pass

    def calc_state_derivative(self, state: State, system: System, u: Quantity) -> np.ndarray:
        # Unpack variables
        vx = state.vx.to_base_units()
        theta = state.theta.to_base_units()
        omega = state.omega.to_base_units()
        b = system.b
        g = system.g
        moi = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com
        x_ddot_coeff = 1/(
            m_pend**2 * l_com**2 * np.cos(theta)**2 / (moi + m_pend*l_com**2) - (m_cart + m_pend)
            )
        theta_ddot_coeff = m_pend * l_com * np.cos(theta) / (
            m_pend**2 * l_com**2 * np.cos(theta)**2 - (m_cart + m_pend) * (moi + m_pend*l_com**2)
            )

        # Compute derivatives
        x_dot = vx
        x_ddot = x_ddot_coeff * (b*vx - m_pend*l_com*omega**2 * np.sin(theta) + (
            m_pend**2 * l_com**2 * g * np.sin(theta) * np.cos(theta)) / (moi + m_pend*l_com**2) - u)
        theta_dot = omega
        theta_ddot = theta_ddot_coeff * (
            -b*vx + m_pend*l_com*omega**2 * np.sin(theta) - (m_cart+m_pend) * g * np.tan(theta) + u
            )

        # Return state derivative
        return self.state_derivative_to_vector(x_dot, x_ddot, theta_dot, theta_ddot)

class BasicDynamics:
    def __init__(self) -> None:
        self.linear = LinearizedModel()
        self.nonlinear = NonlinearModel()
