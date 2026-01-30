from dataclasses import dataclass

import numpy as np
from pint import Quantity

from src import ureg
from src.animate import SimAnimator
from src.primitives import Process, State
from src.sim_components import Cart, Pendulum, System
from src.controllers import AbstractController, ConstantController
from src.dynamics import BasicDynamics

@dataclass
class Simulation(Process):
    system: System
    controller: AbstractController 
    state: State
    dynamics: BasicDynamics = BasicDynamics()
    time: float = 0.0

    def update(self, time: Quantity) -> None:
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time
        
        if delta_time == 0:
            return


        # Input from control law
        u = self.controller.compute_u(self.system, self.state)

        # STATE UPDATE (using nonlinear dynamics)
        self.state = self.dynamics.nonlinear.update_state(self.system, self.state, u, delta_time)

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

    # Time frames for the animation
    times = np.arange(0, 20, 0.001) * ureg.second

    # Create the animator and show the animation
    animator = SimAnimator(system, sim, times)
    animator.show()

if __name__ == "__main__":
    main()