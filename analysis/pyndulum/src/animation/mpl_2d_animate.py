from dataclasses import dataclass

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.animation import mpl_2d_objects as obj
from src.system import System
from src.variables import State


@dataclass
class MplPlotFormatter:
    plot_title: str = "Inverted Pendulum System"
    xlabel: str = "X ({:~P})"
    ylabel: str = "Z ({:~P})"
    axis_units: Quantity = ureg.meter
    axis_equal: bool = False
    grid: bool = True
    limits: tuple[tuple[Quantity]] | None = None
    textbox_format: dict | None = None

    def __post_init__(self) -> None:
        if self.textbox_format is None:
            self.textbox_format = {"facecolor":"w", "alpha":0.5, "pad":5}

    def format_plot(self, ax: plt.Axes, systems: list[System], history: pd.DataFrame) -> None:
        # Set titles
        ax.set_title(self.plot_title)
        ax.set_xlabel(self.xlabel.format(self.axis_units))
        ax.set_ylabel(self.ylabel.format(self.axis_units))

        # Set axis limits and units
        ax.xaxis.set_units(self.axis_units)
        ax.yaxis.set_units(self.axis_units)
        self.set_plot_limits(systems, history)
        ax.set_xlim(self.limits[0])
        ax.set_ylim(self.limits[1])

        # Set plot area characteristics
        if self.axis_equal:
            ax.set_aspect("equal", adjustable="box")
        if self.grid:
            ax.grid()

    def set_plot_limits(self, systems: list[System], history: pd.DataFrame) -> None:
        # Check if limits have already been set
        if self.limits is not None:
            return

        # Initialize limits to be updated based on system bounding boxes
        limits = [[0,0], [0,0]]

        for sys in systems:
            # Calculate the plot limits to always keep the system in view
            bbox_x, _, bbox_z = sys.get_bounding_box(history)
            xlims = bbox_x
            ylims = bbox_z
            # Add margin and make sure the origin is in view
            margin = 0.5 * ureg.meter
            xlims = [min(0, xlims[0]) - margin, max(0, xlims[1]) + margin]
            ylims = [min(0, ylims[0]) - margin, max(0, ylims[1]) + margin]
            # Update based on previously computed limits
            limits = [[min(xlims[0], limits[0][0]), max(xlims[1], limits[0][1])]
                      ,[min(ylims[0], limits[1][0]), max(ylims[1], limits[1][1])]]

        self.limits = limits

class Mpl2dAnimator:
    def __init__(self,
                 plot_formatter: MplPlotFormatter | None = None,
                 sprite_formatter: obj.MplSpriteFormatter | None = None,
                 refresh_rate: Quantity = 30 * ureg.hertz,
                 *args: list,
                 display_eng_info: bool = False) -> None:
        self.plot_formatter = plot_formatter if plot_formatter else MplPlotFormatter()
        self.sprite_formatter = (sprite_formatter if sprite_formatter
                                 else obj.MplSpriteFormatter())
        self.fig, self.ax = plt.subplots()
        self.refresh_rate = refresh_rate
        self.display_eng_info = display_eng_info

    def create_system_animation(self,
                                systems: list[System],
                                times: Quantity,
                                history_df: pd.DataFrame,
                                *args: list,
                                show_progress: bool = True,
                                ) -> None:
        # Assign attributes for ease of access in update func
        self.systems = systems
        self.times = times
        self.history = history_df

        # Format plot
        self.plot_formatter.format_plot(self.ax, systems, history_df)

        # Create animation objects
        self.objects = []
        for sys in self.systems:
            self.objects.extend(self.gen_anim_objects(sys))

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

    def gen_anim_objects(self, sys: System) -> list[obj.AnimObject]:
        sprite_generator = obj.SpriteGenerator(self.sprite_formatter)
        sim_objects: list[obj.AnimObject] = []
        # Cart and pendulum
        sim_objects.append(obj.AnimRectangle(sprite_generator, sys.cart, self.ax))
        sim_objects += obj.AnimCollection(sprite_generator, sys.pendulum, self.ax).patches
        # Text box for time
        sim_objects.append(obj.AnimText("Time: {time:.2f~P}", self.ax, x=0.02, y=0.02,
                                        bbox=self.plot_formatter.textbox_format,
                                        transform=self.ax.transAxes, ha="left"))
        # Additional, optional display items
        if self.display_eng_info:
            # Pendulum centroid
            sim_objects.append(obj.AnimPoint(sprite_generator, sys.pendulum,
                                             sys.pendulum.get_centroid, self.ax))
            # Text boxes for state variables/input and disturbances
            sim_objects.append(obj.AnimText(("x={x:.2f~P}\n"
                                            r"$v_x$={vx:.2f~P}" + "\n"
                                            r"$\theta$={theta:.2f~P}" + "\n"
                                            r"$\omega$={omega:.2f~P}" + "\n"
                                            r"u={u:.2f~P}"),
                                            self.ax, x=0.98, y=0.02,
                                            bbox=self.plot_formatter.textbox_format,
                                            transform=self.ax.transAxes, ha="right"))
            sim_objects.append(obj.AnimText((r"$w_x$={w_x:.2f~P}" + "\n"
                                             r"$w_v$={w_vx:.2f~P}" + "\n"
                                             r"$w_\theta$={w_theta:.2f~P}" + "\n"
                                             r"$w_\omega$={w_omega:.2f~P}"),
                                             self.ax, x=0.98, y=0.98,
                                             bbox=self.plot_formatter.textbox_format,
                                             transform=self.ax.transAxes, ha="right", va="top"))

        # Set validity time interval for each object to control when they are displayed
        for sim_obj in sim_objects:
            sim_obj.times = sys.times

        return sim_objects

    def init_anim(self) -> tuple[Rectangle, plt.Line2D]:
        """Initializes the animation plot (runs once at the start)."""
        artists = tuple(obj.initialize() for obj in self.objects)
        # Must return an iterable of artists when blit=True
        return artists

    # Updates the graph
    def update(self, frame: float) -> tuple:
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
