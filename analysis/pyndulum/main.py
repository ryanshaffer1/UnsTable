import cProfile
import logging
import logging.config
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.animation import SimAnimator
from src.controllers import AbstractController, ConstantController, LQRController
from src.dynamics import BasicDynamics
from src.integrators import Integrator, RK4Integrator
from src.outputs import record_outputs
from src.primitives import Input, State
from src.system import Actuator, Cart, Pendulum, System
from src.utils import LOGGING_CONFIG

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

@dataclass
class Simulation:
    system: System
    controller: AbstractController
    state: State
    dynamics: BasicDynamics = field(default_factory=BasicDynamics)
    integrator: Integrator = field(default_factory=RK4Integrator)
    time: Quantity = 0.0 * ureg.second

    def run(self, times: np.ndarray, *args: tuple, show_progress: bool=True) -> list[State]:
        # Initialize output storage
        states = np.ndarray([4, len(times)], dtype=float)
        inputs = np.ndarray([len(times)], dtype=float)

        # Initialize control input at zero
        self.u = 0 * ureg.newton
        self.lag_u = 0 * ureg.newton
        # Initialize lag window to simulation start time
        self.lag_window_start = times[0]

        # Wrap times in tqdm to show progress bar
        if show_progress:
            times = tqdm(times)

        # Iterate through all simulation times
        for i, t in enumerate(times):
            # Update state and control input
            self.update(t)

            # Add time step to state history and input history
            states[:,i] = self.state.to_vector()
            inputs[i] = self.u.to_base_units().magnitude

        return states, inputs

    def update(self, time: Quantity) -> None:
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time

        # Update the commanded actuator input based on control law
        if self.system.actuator.is_update_time(time, delta_time):
            self.lag_u = self.controller.compute_u(self.system, self.state)
            self.lag_window_start = self.time
        # Actually apply the new actuator input if the lag window is satisfied
        if self.system.actuator.is_past_lag_time(time, self.lag_window_start):
            self.u = self.lag_u

        # STATE UPDATE (using numerical integrator and nonlinear dynamics)
        self.state = self.integrator.step(self.dynamics.nonlinear.calc_state_derivative,
                                          self.state,
                                          delta_time.magnitude,
                                          self.system,
                                          self.u)

def main() -> None:
    # Initialize simulation
    init_state = State(x = 0 * ureg.meter,
                       vx = 1 * ureg.meter/ureg.second,
                       theta = 0 * ureg.degree,
                       omega = 0 * ureg.radian/ureg.second)
    system = System(Actuator(),
                    Cart(mass=5*ureg.kg, friction_coeff=1*ureg.newton*ureg.second/ureg.meter),
                    Pendulum(mass=1*ureg.kg, length=2*ureg.meter),
                    gravity=10*ureg.meter/ureg.second**2)
    system = System(Actuator(force_limit=50*ureg.newton,
                             refresh_rate=20*ureg.hertz,
                             command_lag=0.002*ureg.second,
                             ),
                    Cart(), Pendulum())
    controller = ConstantController(0 * ureg.newton)
    controller = LQRController(
        Q=np.array([[10,0,0,0],[0,1,0,0],[0,0,100,0],[0,0,0,10]]),
        R=1,
        system=system,
        setpoint=[0,0,0,0])
    sim = Simulation(system, controller, init_state)

    # Output flags
    show_progress = True
    save_animation = False
    show_animation = True

    # Time frames for the animation
    times = np.arange(0, 10, 0.001) * ureg.second

    # Run the simulation
    logger.info("Running simulation...")
    states, inputs = sim.run(times, show_progress=show_progress)
    histories = {"states": states, "inputs": inputs} # TODO: replace with output_df

    # Build output dataframe and record output metrics
    output_df = pd.concat((State.history_to_dataframe(states, times),
                           Input.history_to_dataframe(inputs, times)),
                          axis=1)
    record_outputs(output_df, step_variable="vx")


    # Create the animator and show the animation
    animator = SimAnimator(system, times, histories, show_progress=show_progress)

    if save_animation:
        logger.info("Saving animation...")
        animator.save("pyndulum.gif")

    if show_animation:
        logger.info("Showing animation...")
        animator.show()

if __name__ == "__main__":
    profiling = "--profile" in sys.argv[1:]

    if profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    main()

    if profiling:
        profiler.disable()
        profiler.dump_stats("profile.dat")
