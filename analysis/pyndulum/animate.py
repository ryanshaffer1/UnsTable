import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

class SimAnimator:
    def __init__(self, simulation, frames):
        self.sim = simulation
        self.fig, self.ax = plt.subplots()

        # Format plot
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-8, 12)
        self.ax.set_xlabel("X-axis")
        self.ax.set_ylabel("Y-axis")
        self.ax.set_title("Inverted Pendulum System")
        self.ax.grid(True)
        # Add labels for time and state
        self.time_label = self.ax.text(0.02,0.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=self.ax.transAxes, ha="left")
        self.state_label = self.ax.text(0.98,0.02, "", bbox={'facecolor':'w', 'alpha':0.5, 'pad':5}, transform=self.ax.transAxes, ha="right")

        # Add cart as a recangle patch
        self.cart = self.ax.add_patch(Rectangle((0,0), self.sim.cart.width, self.sim.cart.height))
        # Add pendulum as a line
        self.pendulum, = self.ax.plot([], [], lw=2*self.sim.pendulum.mass, color='orange')

        # Create animation object with callbacks to initialize and update methods
        self.ani = FuncAnimation(
            self.fig, 
            func=self.update, 
            frames=frames, 
            init_func=self.init_anim, 
            blit=True, # Blitting optimizes drawing
        )

    def init_anim(self):
        """Initializes the animation plot (runs once at the start)."""
        self.cart.set_xy(([],[]))
        self.pendulum.set_data([],[])
        # Must return an iterable of artists when blit=True
        return (self.cart, self.pendulum)

    # updates the data and graph
    def update(self, time):
        self.sim.update(time)
        x = self.sim.state.x

        # Updating the simulation objects
        self.cart.set_xy(self.sim.cart.get_ll_corner(x))
        bob_position = self.sim.pendulum.get_tip_position(x, self.sim.cart.y_top, self.sim.state.theta)
        self.pendulum.set_data([x, bob_position[0]], [self.sim.cart.y_top, bob_position[1]])

        # Update label
        self.time_label.set_text(f"Time: {time:.2f} seconds")
        self.state_label.set_text(rf"""x={self.sim.state.x:.2f}
$v_x$={self.sim.state.vx:.2f}
$\theta$={self.sim.state.theta:.2f}
$\omega$={self.sim.state.omega:.2f}""")

        return (self.cart, self.pendulum, self.time_label, self.state_label)

    def show(self):
        """Displays the animation."""
        plt.show()
