from dataclasses import dataclass
import numpy as np
from animate import SimAnimator

@dataclass
class State:
    x: float
    vx: float
    theta: float
    omega: float

@dataclass
class Cart:
    mass: float = 1
    y_top: float = 0
    width: float = 4
    height: float = 2
    
    def get_ll_corner(self, x_center):
        return (x_center - self.width / 2, self.y_top - self.height)

@dataclass
class Pendulum:
    mass: float = 3
    length: float = 8

    def get_tip_position(self, cart_x, cart_y, theta):
        bob_x = cart_x + self.length * np.sin(theta)
        bob_y = cart_y + self.length * np.cos(theta)
        return (bob_x, bob_y)

class Simulation:
    def __init__(self, init_state: State, cart: Cart, pendulum: Pendulum, init_time: float = 0.0):
        self.state = init_state
        self.cart = cart
        self.pendulum = pendulum
        self.time = init_time

    def update(self, time):
        # Update sim time and track time step
        prev_time = self.time
        self.time = time
        delta_time = time - prev_time

        # Physics update (simple example)
        self.state.x = self.state.x + self.state.vx * delta_time
        self.state.theta = self.state.theta + self.state.omega * delta_time

def main():
    # Initialize simulation
    init_state = State(x=0, vx=0.5, theta=0, omega=5*np.pi/180)
    sim = Simulation(init_state, Cart(), Pendulum())

    # Time frames for the animation
    times = np.arange(0, 10, 0.5)

    # Create the animator and show the animation
    animator = SimAnimator(sim, times)
    animator.show()

if __name__ == "__main__":
    main()