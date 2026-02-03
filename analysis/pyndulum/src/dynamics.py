from abc import ABC, abstractmethod

import numpy as np
from pint import Quantity

from src import ureg
from src.primitives import State
from src.system import System


class AbstractDynamicsModel(ABC):
    @abstractmethod
    def calc_state_derivative(self, state: State, system: System, u: Quantity) -> np.ndarray:
        pass


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

        A, B = self.get_A_B_unitless(m_cart, m_pend, l_com, moi, g, b)
        return A, B

    # Check, then strip, units from the equations of motion
    @ureg.wraps(None,
                (None,
                 ureg.kilogram,
                 ureg.kilogram,
                 ureg.meter,
                 ureg.kilogram*ureg.meter**2,
                 ureg.meter/ureg.second**2,
                 ureg.newton*ureg.second/ureg.meter,
                 ), strict=True)
    def get_A_B_unitless(self, # noqa: PLR0913
                         m_cart: float,
                         m_pend: float,
                         l_com: float,
                         moi: float,
                         g: float,
                         b: float,
                         ) -> tuple[np.ndarray, np.ndarray]:
        # Denominators for the A and B matrix elements
        p = m_cart + m_pend - (m_pend**2 * l_com**2) / (moi + m_pend * l_com**2)
        q = (m_pend**2 * l_com**2 - (moi + m_pend * l_com**2) * (m_cart + m_pend))/(m_pend*l_com)

        # Calculate the non-trivial matrix and vector elements
        a22 = (-b / p)
        a32 = (-g * m_pend**2 * l_com**2 / (p * (moi + m_pend * l_com**2)))
        a24 = (-b / q)
        a34 = ((-1 * (m_cart + m_pend) * g)/q)

        b2 = (1/p)
        b4 = (1/q)

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
        vx = state.vx
        theta = state.theta
        omega = state.omega
        b = system.b
        g = system.g
        moi = system.moi_pend
        m_cart = system.m_cart
        m_pend = system.m_pend
        l_com = system.l_com

        # Solve equations of motion with units checked and stripped
        x_dot, x_ddot, theta_dot, theta_ddot = self.calc_state_derivative_unitless(
            vx, theta, omega, m_cart, m_pend, l_com, moi, g, b, u,
        )

        # Return state derivative as array
        return np.array([x_dot, x_ddot, theta_dot, theta_ddot])

    # Check, then strip, units from the equations of motion
    @ureg.wraps(None,
                (None,
                 ureg.meter/ureg.second,
                 ureg.radian,
                 ureg.radian/ureg.second,
                 ureg.kilogram,
                 ureg.kilogram,
                 ureg.meter,
                 ureg.kilogram*ureg.meter**2,
                 ureg.meter/ureg.second**2,
                 ureg.newton*ureg.second/ureg.meter,
                 ureg.newton,
                 ), strict=True)
    def calc_state_derivative_unitless(self, # noqa: PLR0913
                                       vx: float,
                                       theta: float,
                                       omega: float,
                                       m_cart: float,
                                       m_pend: float,
                                       l_com: float,
                                       moi: float,
                                       g: float,
                                       b: float,
                                       u: float,
                                       ) -> tuple[float, float, float, float]:
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
            -b*vx + m_pend*l_com*omega**2 * np.sin(theta) - (m_cart+ m_pend) * g * np.tan(theta) + u
            )
        return x_dot, x_ddot, theta_dot, theta_ddot


class BasicDynamics:
    def __init__(self) -> None:
        self.linear = LinearizedModel()
        self.nonlinear = NonlinearModel()
