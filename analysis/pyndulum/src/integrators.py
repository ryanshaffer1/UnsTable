from abc import ABC, abstractmethod
from collections.abc import Callable

from src.primitives import State


class Integrator(ABC):
    @abstractmethod
    def step(self, func: Callable, state: State, dt: float, *args: tuple, **kwargs: dict) -> State:
        pass


class EulerIntegrator(Integrator):
    def step(self, func: Callable, state: State, dt: float, *args: tuple, **kwargs: dict) -> State:

        # Get state derivative from func
        state_derivative = func(state, *args, **kwargs)

        # Euler integration to get new state
        new_state = state.add_vector(state_derivative * dt)
        return new_state


class RK4Integrator(Integrator):
    def step(self, func: Callable, state: State, dt: float, *args: tuple, **kwargs: dict) -> State:
        # Compute k1
        k1 = func(state, *args, **kwargs)

        # Compute k2
        state_k2 = state.add_vector(0.5 * k1 * dt)
        k2 = func(state_k2, *args, **kwargs)

        # Compute k3
        state_k3 = state.add_vector(0.5 * k2 * dt)
        k3 = func(state_k3, *args, **kwargs)

        # Compute k4
        state_k4 = state.add_vector(k3 * dt)
        k4 = func(state_k4, *args, **kwargs)

        # Combine to get new state
        new_state = state.add_vector((dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4))
        return new_state
