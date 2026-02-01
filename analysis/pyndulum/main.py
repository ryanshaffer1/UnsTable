from dataclasses import dataclass

import numpy as np
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.animate import SimAnimator
from src.primitives import State
from src.system import Cart, Pendulum, System
from src.controllers import AbstractController, ConstantController
from src.dynamics import BasicDynamics
from src.integrators import Integrator, RK4Integrator

@dataclass
class Simulation:
    system: System
    controller: AbstractController 
    state: State
    dynamics: BasicDynamics = BasicDynamics()
    integrator: Integrator = RK4Integrator()
    time: Quantity = 0.0 * ureg.second

    def run(self, times: np.ndarray) -> list[State]:
        # Initialize output storage
        states = np.ndarray([4, len(times)], dtype=float)
        inputs = np.ndarray([len(times)], dtype=float)

        # Iterate through all simulation times
        for i, t in enumerate(times):
            # Update state and control input
            u = self.update(t)

            # Add time step to state history and input history
            states[:,i] = self.state.to_vector()
            inputs[i] = u.magnitude
            
        return states, inputs

    def update(self, time: Quantity) -> Quantity:
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time
        if delta_time.magnitude == 0:
            return 0 * ureg.newton  # No time has passed, no update needed

        # Input from control law
        u = self.controller.compute_u(self.system, self.state)

        # STATE UPDATE (using numerical integrator and nonlinear dynamics)
        self.state = self.integrator.step(self.dynamics.nonlinear.calc_state_derivative, 
                                          self.state,
                                          delta_time.magnitude,
                                          self.system, 
                                          u)

        return u

def main():
    # Initialize simulation
    init_state = State(x = 0 * ureg.meter, 
                       vx = 0 * ureg.meter/ureg.second, 
                       theta = 0 * ureg.degree, 
                       omega = .5 * ureg.radian/ureg.second)
    system = System(Cart(mass=5*ureg.kg, friction_coeff=1*ureg.newton*ureg.second/ureg.meter), 
                    Pendulum(mass=1*ureg.kg, length=2*ureg.meter),
                    gravity=10*ureg.meter/ureg.second**2)
    controller = ConstantController(0 * ureg.newton)
    sim = Simulation(system, controller, init_state)
    
    # Output flags
    save_animation = False
    show_animation = True

    # Time frames for the animation
    times = np.arange(0, 10, 0.01) * ureg.second

    # Run the simulation
    print("Running simulation...")
    states, inputs = sim.run(tqdm(times))

    # Create the animator and show the animation
    animator = SimAnimator(system, times, {"states": states, "inputs": inputs})

    if save_animation:
        print("\nSaving animation...")
        animator.save("pyndulum.gif")

    if show_animation:
        print("\nShowing animation...")
        animator.show()
    
if __name__ == "__main__":
    main()