import numpy as np

from src import ureg
from src.animate import SimAnimator
from src.primitives import Process, State
from src.sim_components import Cart, Pendulum, Plant, System
from src.controllers import AbstractController, ConstantController

class Simulation(Process):
    def __init__(self, 
                 system: System,
                 controller: AbstractController, 
                 init_state: State, 
                 plant: Plant = Plant(), 
                 init_time: float = 0.0):
        self.system = system
        self.controller = controller
        self.state = init_state
        self.plant = plant
        self.time = init_time

    def update(self, time: float) -> None:
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time
        
        if delta_time == 0:
            return

        # STATE UPDATE
        # Construct A and B matrices from plant
        A, B = self.plant.get_A_B(self.system)

        # Input from control law
        u = 0 #* ureg.newton  # No input force
        u = self.controller.compute_u(self.system, self.state)

        # Compute state derivative
        state_deriv = A @ self.state.to_vector() + B * u
        x_dot = state_deriv[0] * ureg.meter / ureg.second
        x_ddot = state_deriv[1] * ureg.meter / ureg.second**2
        theta_dot = state_deriv[2] * ureg.radian / ureg.second
        theta_ddot = state_deriv[3] * ureg.radian / ureg.second**2

        # Euler integration to get new state
        new_state = State(
            x = self.state.x + x_dot * delta_time,
            vx = self.state.vx + x_ddot * delta_time,
            theta = self.state.theta + theta_dot * delta_time,
            omega = self.state.omega + theta_ddot * delta_time
        )
        self.state = new_state

def main():
    # Initialize simulation
    init_state = State(x = 0 * ureg.meter, 
                       vx = 0.1 * ureg.meter/ureg.second, 
                       theta = 0 * ureg.degree, 
                       omega = 5 * ureg.degree/ureg.second)
    system = System(Cart(), Pendulum())
    controller = ConstantController(0 * ureg.newton)
    sim = Simulation(system, controller, init_state)

    # Time frames for the animation
    times = np.arange(0, 10, 0.01) * ureg.second

    # Create the animator and show the animation
    animator = SimAnimator(system, sim, times)
    animator.show()

if __name__ == "__main__":
    main()