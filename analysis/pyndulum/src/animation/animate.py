import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.animation import objects
from src.primitives import State
from src.system import System


def basic_objects(ax: plt.Axes, sys: System, textbox_format: dict) -> list[objects.AnimObject]:
    # Add animated objects and labels
    cart_anim = objects.AnimRectangle(sys.cart, ax, color="gray")
    pend_base_anim = objects.AnimLine(sys.pendulum, ax,
                                        lw=max(1, sys.pendulum.thickness.magnitude),
                                        color="brown")
    pend_top_anim = objects.OffsetAnimLine(pend_base_anim, ax,
                                           offset=("end",0*ureg.meter,0*ureg.meter,90*ureg.degree),
                                           length=sys.pendulum.length, # TODO make this variable
                                           lw=max(1, sys.pendulum.thickness.magnitude),
                                           color="saddlebrown")
    time_text_anim = objects.AnimText("Time: {time:.2f~P}", ax, x=0.02, y=0.02,
                                      bbox=textbox_format, transform=ax.transAxes,
                                      ha="left")
    state_text_anim = objects.AnimText(("x={x:.2f~P}\n"
                                        r"$v_x$={vx:.2f~P}" + "\n"
                                        r"$\theta$={theta:.2f~P}" + "\n"
                                        r"$\omega$={omega:.2f~P}" + "\n"
                                        r"u={u:.2f~P}"),
                                        ax, x=0.98, y=0.02, bbox=textbox_format,
                                        transform=ax.transAxes, ha="right")
    dist_text_anim = objects.AnimText((r"$w_x$={w_x:.2f~P}" + "\n"
                                       r"$w_v$={w_vx:.2f~P}" + "\n"
                                       r"$w_\theta$={w_theta:.2f~P}" + "\n"
                                       r"$w_\omega$={w_omega:.2f~P}"),
                                       ax, x=0.98, y=0.98, bbox=textbox_format,
                                       transform=ax.transAxes, ha="right", va="top")

    sim_objects = [cart_anim,
               pend_base_anim,
               pend_top_anim,
               time_text_anim,
               state_text_anim,
               dist_text_anim,
               ]
    return sim_objects

class SimAnimator:
    def __init__(self,
                 system: System,
                 times: Quantity,
                 history_df: pd.DataFrame,
                 refresh_rate: Quantity = 30 * ureg.hertz,
                 *args: tuple,
                 show_progress: bool = True,
                 ) -> None:
        self.sys = system
        self.history = history_df
        self.times = times
        self.fig, self.ax = plt.subplots()
        self.refresh_rate = refresh_rate
        self.format_plot()

        self.objects = basic_objects(self.ax, self.sys, self.textbox_format)

        # Calculate frames to control animation refresh rate
        sim_time_step = times[1] - times[0]
        steps_per_frame = max(1, int(1 / self.refresh_rate / sim_time_step))
        frames = range(0, len(times), steps_per_frame)

        # Wrap times in tqdm to show progress bar
        if show_progress:
            frames = tqdm(frames)

        # Create animation object with callbacks to initialize and update methods
        self.ani = FuncAnimation(
            self.fig,
            func=self.update,
            frames=frames,
            init_func=self.init_anim,
            blit=True, # Blitting optimizes drawing
            interval = int((1/self.refresh_rate).to("millisecond").magnitude),
        )

    def save(self, filename: str = "animation.gif") -> None:
        self.ani.save(filename, writer="pillow", fps=self.refresh_rate.magnitude)

    def format_plot(self) -> None:
        # Calculate plot limits to keep cart/pendulum in view
        endpoint_history = self.sys.trace_pend_endpoint_history(self.history)
        xlims = (np.min(endpoint_history[["base_x","tip_x"]]),
                 np.max(endpoint_history[["base_x","tip_x"]]))
        ylims = (np.min(endpoint_history[["base_y","tip_y"]]),
                 np.max(endpoint_history[["base_y","tip_y"]]))
        # Add margin and make sure origin is in view
        margin = 0.5 * ureg.meter
        xlims = [min(0, xlims[0]) - margin, max(0, xlims[1]) + margin]
        ylims = [min(0, ylims[0]) - margin, max(0, ylims[1]) + margin]

        # Format plot
        self.ax.xaxis.set_units(ureg.meter)
        self.ax.yaxis.set_units(ureg.meter)
        self.ax.set_xlim(*xlims)
        self.ax.set_ylim(*ylims)
        self.ax.set_xlabel(f"X ({self.ax.xaxis.get_units():~P})")
        self.ax.set_ylabel(f"Y ({self.ax.yaxis.get_units():~P})")
        self.ax.set_title("Inverted Pendulum System")
        self.ax.grid()

        # Set format for text boxes
        self.textbox_format = {"facecolor":"w", "alpha":0.5, "pad":5}

    def init_anim(self) -> tuple[Rectangle, plt.Line2D]:
        """Initializes the animation plot (runs once at the start)."""
        artists = tuple(obj.initialize() for obj in self.objects)
        # Must return an iterable of artists when blit=True
        return artists

    # Updates the graph
    def update(self, frame: float) -> tuple[Rectangle, plt.Line2D, plt.Text, plt.Text]:
        time = self.times[frame]
        sim_info = self.history.loc[time].to_dict()
        state_dict = {var: sim_info.pop(var) for var in State.get_variable_names()}
        state = State(**state_dict)

        # Updating the simulation objects
        artists = tuple(obj.update(state, time, **sim_info) for obj in self.objects)
        return artists

    def show(self) -> None:
        """Displays the animation."""
        plt.show()
