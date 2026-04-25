import argparse
import cProfile
import logging
import logging.config
import sys
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.controllers import AbstractController
from src.dynamics import BasicDynamics
from src.integrators import Integrator, RK4Integrator
from src.outputs import record_outputs
from src.system import System
from src.utils import LOGGING_CONFIG, add_yaml_constructors
from src.variables import Disturbance, GaussianDisturbance, Input, State

# Set up logger
logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("log")

@dataclass
class Simulation:
    systems: list[System]
    controller: AbstractController
    state: State
    dynamics: BasicDynamics = field(default_factory=BasicDynamics)
    integrator: Integrator = field(default_factory=RK4Integrator)
    disturbances: Disturbance = field(default_factory=GaussianDisturbance)
    time: Quantity = 0.0 * ureg.second
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(seed=42))

    def __post_init__(self) -> None:
        # Iterate over all defined fields in the dataclass
        for f in fields(self):
            # Check if the current value is None and if a default_factory was specified
            if getattr(self, f.name) is None and f.default_factory is not MISSING:
                # If both are true, call the default_factory and set the attribute
                setattr(self, f.name, f.default_factory())

    def run(self, times: np.ndarray, *args: tuple, show_progress: bool=True) -> list[State]:
        # Initialize output storage
        states = np.ndarray([4, len(times)], dtype=float)
        inputs = np.ndarray([len(times)], dtype=float)
        disturbances = np.ndarray([4, len(times)], dtype=float)

        # Initialize control input at zero
        self.u = 0 * ureg.newton
        self.lag_u = 0 * ureg.newton
        # Initialize lag window to simulation start time
        self.lag_window_start = times[0]

        # Wrap times in tqdm to show progress bar
        if show_progress:
            times = tqdm(times)

        # Keep track of active system
        self.active_system = self.get_valid_system(time = 0*ureg.seconds)

        # Iterate through all simulation times
        for i, t in enumerate(times):
            # Update state and control input
            self.update(t)

            # Add time step to state history and input history
            states[:,i] = self.state.to_vector()
            inputs[i] = self.u.to_base_units().magnitude
            disturbances[:,i] = self.dist

        return states, inputs, disturbances

    def update(self, time: Quantity) -> None:
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time

        # Get active system
        system = self.get_valid_system(time)

        # Handle transitions from one active system to another
        if system is not self.active_system:
            self.state = system.update_state_during_transition(self.active_system, self.state)
            self.active_system = system

        # Update the commanded actuator input based on control law
        if system.actuator.is_update_time(time, delta_time):
            self.lag_u = self.controller.compute_u(system, self.state)
            self.lag_window_start = self.time
        # Actually apply the new actuator input if the lag window is satisfied
        if system.actuator.is_past_lag_time(time, self.lag_window_start):
            self.u = self.lag_u

        # STATE UPDATE (using numerical integrator and nonlinear dynamics)
        self.state = self.integrator.step(self.dynamics.nonlinear.calc_state_derivative,
                                          self.state,
                                          delta_time.magnitude,
                                          system,
                                          self.u)

        # Add disturbances to the state
        self.dist = self.disturbances.apply(self.state, time, self.rng)
        self.state = self.state.add_vector(self.dist)

    def get_valid_system(self, time: Quantity) -> System:
        valid_systems = [sys for sys in self.systems if sys.valid_time(time)]
        if not valid_systems or len(valid_systems) > 1:
            msg = f"Time {time} has {len(valid_systems)} valid systems. Check system time intervals."
            logger.warning(msg)
            raise ValueError(msg)
        return valid_systems[0]


def main(parameter_file: Path) -> None:
    add_yaml_constructors()

    with parameter_file.open("r") as stream:
        params = yaml.safe_load(stream)

    # Unpack required parameters from YAML file
    systems = params["systems"] if "systems" in params else params["system"]
    controller = params["controller"]
    init_state = params["initial_state"]
    times = params["times"]
    animator = params.get("animation")
    settings = params.get("settings", {})

    # Set validity intervals for system if only one is provided
    if isinstance(systems, System):
        systems.times = (times[0], times[-1])
        systems = [systems]


    # Set random seed
    rng = np.random.default_rng(settings.get("random_seed", 42))

    # Initialize simulation
    sim = Simulation(systems=systems,
                     controller=controller,
                     state=init_state,
                     dynamics=params.get("dynamics"),
                     integrator=params.get("integrator"),
                     disturbances=params.get("disturbances"),
                     rng=rng,
                     )

    # Run the simulation
    logger.info("Running simulation...")
    states, inputs, disturbances = sim.run(times, show_progress=settings.get("show_progress", True))

    # Build output dataframe and record output metrics
    output_df = pd.concat((State.history_to_dataframe(states, times),
                           Input.history_to_dataframe(inputs, times),
                           Disturbance.history_to_dataframe(disturbances, times)),
                          axis=1)
    record_outputs(output_df, step_variable="vx")


    # Create the animator and show the animation
    logger.info("Setting up animation...")
    if animator:
        animator.create_system_animation(systems,
                                         times,
                                         output_df,
                                         show_progress=settings.get("show_progress", True))

        if settings.get("save_animation", False):
            logger.info("Saving animation...")
            animator.save("pyndulum.gif")

        if settings.get("show_animation", True):
            logger.info("Showing animation...")
            animator.show()

def cli(args: list[str]) -> tuple[Path]:
    parser = argparse.ArgumentParser(description="Simulate and animate a cart-pendulum system.")
    parser.add_argument("--parameter-file", type=Path, default=Path("inputs/simple.yaml"),
                        help="Path to YAML file containing system parameters.")
    return (parser.parse_args(args).parameter_file,)

if __name__ == "__main__":
    profiling = "--profile" in sys.argv[1:]

    if profiling:
        sys.argv.remove("--profile")
        profiler = cProfile.Profile()
        profiler.enable()

    main(*cli(sys.argv[1:]))

    if profiling:
        profiler.disable()
        profiler.dump_stats("profile.dat")
