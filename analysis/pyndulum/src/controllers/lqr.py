import numpy as np
from pint import Quantity
from scipy.linalg import solve_continuous_are

from src import ureg
from src.controllers.abstract_controller import AbstractController
from src.dynamics import LinearizedModel
from src.primitives import State
from src.system import System

class LQRController(AbstractController):
    def __init__(self,
                 Q: np.ndarray,
                 R: np.ndarray | float,
                 system: System,
                 linear_dynamics: LinearizedModel = LinearizedModel(),
                 setpoint: np.ndarray = np.zeros(4),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.Q = Q
        self.R = R
        self.setpoint = setpoint

        # Ensure that R is a 2dmatrix
        if isinstance(self.R, float | int):
            self.R = np.array([self.R]).reshape(1, 1)
        
        self.K = self.calc_k(linear_dynamics, system)

    def calc_k(self, linear_dynamics: LinearizedModel, system: System):
        # Get A and B from linear dynamics
        A, B = linear_dynamics.get_A_B(system)
        
        # Solve the algebraic ricatti equation
        P = solve_continuous_are(A, B, self.Q, self.R)

        # Calculate K
        K = np.linalg.inv(self.R) @ B.T @ P
        return K

    def compute_u(self, system: System, state: State) -> Quantity:
        # Proportional control law: u = -K * x
        u = (-self.K @ (state.to_vector() - self.setpoint))
        u_requested = u[0] * ureg.newton
        
        # Enforce actuator limit
        u = system.actuator.enforce_limit(u_requested)
        return u
