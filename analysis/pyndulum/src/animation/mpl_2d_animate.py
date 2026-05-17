from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Patch, Rectangle
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.system import Block, BodyRefPoint, Cylinder, RigidBody, RigidBodySystem, Sphere, System
from src.variables import State


class MplSpriteFormatter:
    def __init__(self, **format_spec: dict) -> None:
        self.format_spec = format_spec
        self.specs_by_name = self.validate_spec_input()

    def validate_spec_input(self) -> bool:
        # Check if specs are provided on a name-by-name basis
        # (true if format spec is a dict of dicts)
        if all(isinstance(x, dict) for x in self.format_spec.values()):
            return True
        # If format spec is has some values that are dicts, then something is wrong
        if any(isinstance(x, dict) for x in self.format_spec.values()):
            msg = "MplSpriteFormatter provided invalid input."
            raise ValueError(msg)
        # If format spec is a "raw" dict, then specs are not provided on a name-by-name basis
        return False

    def get_spec(self, name: str) -> dict:
        if self.specs_by_name:
            # Return the format spec corresponding with the object name, or a default, or none
            if name in self.format_spec:
                return self.format_spec[name]
            return self.format_spec.get("default", {})
        # If there is just one format spec, return that
        return self.format_spec

class SpriteGenerator:
    def __init__(self, formatter: MplSpriteFormatter) -> None:
        self.formatter = formatter

    def get_sprite(self, source: RigidBody, format_spec: dict | None = None) -> Patch:
        # Set formatting
        format_spec = format_spec or {}
        format_spec.update(self.formatter.get_spec(source.name))

        match source:
            case Block():
                return Rectangle((0,0), source.width, source.height, **format_spec)
            case Cylinder():
                return Rectangle((0,0), 2*source.radius, source.length, **format_spec)
            case Sphere():
                return Circle((0,0), source.radius, **format_spec)

class AnimObject(ABC):
    def __init__(self, times: tuple[Quantity] | None = None, **kwargs: dict) -> None:
        self.times = times

    @abstractmethod
    def initialize(self) -> plt.Artist:
        pass
    @abstractmethod
    def update(self, state: State, time: Quantity) -> plt.Artist:
        pass

    def valid_time(self, time: Quantity) -> bool:
        return (self.times is None) or (self.times[0] <= time < self.times[1])

class AnimCollection:
    def __init__(self,
                 sprite_gen: SpriteGenerator,
                 source: RigidBodySystem,
                 ax: plt.Axes,
                 collection: Self | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.patches = []
        self.source = source
        self.collection = collection

        # Get collection's format spec to apply to all bodies
        self.format_spec = collection.format_spec if collection else {}
        self.format_spec.update(sprite_gen.formatter.get_spec(self.source.name))

        for body in source.bodies:
            match body:
                case Block():
                    self.patches.append(AnimRectangle(sprite_gen,
                                                      body,
                                                      ax,
                                                      collection=self))
                case Cylinder():
                    self.patches.append(AnimRectangle(sprite_gen,
                                                      body,
                                                      ax,
                                                      collection=self))
                case Sphere():
                    self.patches.append(AnimCircle(sprite_gen,
                                                   body,
                                                   ax,
                                                   collection=self))
                case RigidBodySystem():
                    subsystem = AnimCollection(sprite_gen, body, ax, collection=self)
                    self.patches += subsystem.patches
                case _:
                    msg = f"Unsupported rigid body type for animation: {type(body)}"
                    raise NotImplementedError(msg)

    def update_object(self, obj: RigidBody, state: State) -> None:
        self.source.update_frame(state, obj)

class AnimRectangle(AnimObject):
    def __init__(self,
                 sprite_gen: SpriteGenerator,
                 source: Block,
                 ax: plt.Axes,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        sprite = sprite_gen.get_sprite(self.source, format_spec=prev_format_spec)
        self.sprite = ax.add_patch(sprite)

    def initialize(self) -> Rectangle:
        self.sprite.set_xy(([],[]))
        return self.sprite

    def update(self, state: State, time: Quantity, *args: tuple, **kwargs: dict) -> Rectangle:
        if not self.valid_time(time):
            self.sprite.set_visible(False)
            return self.sprite
        self.sprite.set_visible(True)

        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        lll_corner = self.source.get_point(BodyRefPoint.MINX_MINY_MINZ, cs_type="global")
        rotation = self.source.get_frame_rotation("Y")
        self.sprite.set_angle(-rotation.magnitude)
        self.sprite.set_xy((lll_corner.x, lll_corner.z))
        return self.sprite

class AnimCircle(AnimObject):
    def __init__(self,
                 sprite_gen: SpriteGenerator,
                 source: Sphere,
                 ax: plt.Axes,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        sprite = sprite_gen.get_sprite(self.source, format_spec=prev_format_spec)
        self.sprite = ax.add_patch(sprite)

    def initialize(self) -> Circle:
        self.sprite.set_center(([],[]))
        return self.sprite

    def update(self, state: State, time: Quantity, *args: tuple, **kwargs: dict) -> Circle:
        if not self.valid_time(time):
            self.sprite.set_visible(False)
            return self.sprite
        self.sprite.set_visible(True)

        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        center = self.source.get_point(BodyRefPoint.CENTER, cs_type="global")
        self.sprite.set_center((center.x, center.z))
        return self.sprite

class AnimPoint(AnimObject):
    def __init__(self,
                 sprite_gen: SpriteGenerator,
                 source: RigidBody,
                 update_func: Callable,
                 ax: plt.Axes,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.update_func = update_func
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        # Create a "virtual" sphere to show the point
        point_sphere = Sphere(name=kwargs.get("name","point"),
                              radius=0.5*ureg.inch,
                              mass=0*ureg.kg,
                              body_frame=None,
                              origin_type=BodyRefPoint.CENTER)
        sprite = sprite_gen.get_sprite(point_sphere, format_spec=prev_format_spec)
        self.sprite = ax.add_patch(sprite)

    def initialize(self) -> Circle:
        self.sprite.set_center(([],[]))
        return self.sprite

    def update(self, state: State, time: Quantity, *args: tuple, **kwargs: dict) -> Circle:
        if not self.valid_time(time):
            self.sprite.set_visible(False)
            return self.sprite
        self.sprite.set_visible(True)

        self.source.update_frame(state)
        center = self.update_func()
        self.sprite.set_center((center.x, center.z))
        return self.sprite

class AnimText(AnimObject):
    def __init__(self, fmt: str, ax: plt.Axes, x: float, y: float, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.fmt = fmt
        self.text = ax.text(x, y, "", **kwargs)

    def initialize(self) -> plt.Text:
        self.text.set_text("")
        return self.text

    def update(self, state: State, time: Quantity, *args: tuple, **kwargs: dict) -> plt.Text:
        if not self.valid_time(time):
            self.text.set_visible(False)
            return self.text

        self.text.set_visible(True)

        # Convert state values to display units
        display_state = state.to_display_units()

        text_vars = {**display_state, "time": time, **kwargs}
        content = self.fmt.format(**text_vars)
        self.text.set_text(content)
        return self.text


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
                 sprite_formatter: MplSpriteFormatter | None = None,
                 refresh_rate: Quantity = 30 * ureg.hertz,
                 *args: list,
                 display_eng_info: bool = False) -> None:
        self.plot_formatter = plot_formatter or MplPlotFormatter()
        self.sprite_formatter = sprite_formatter or MplSpriteFormatter()
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

    def gen_anim_objects(self, sys: System) -> list[AnimObject]:
        sprite_generator = SpriteGenerator(self.sprite_formatter)
        sim_objects: list[AnimObject] = []
        # Cart and pendulum
        sim_objects.append(AnimRectangle(sprite_generator, sys.cart, self.ax))
        sim_objects += AnimCollection(sprite_generator, sys.pendulum, self.ax).patches
        # Text box for time
        sim_objects.append(AnimText("Time: {time:.2f~P}", self.ax, x=0.02, y=0.02,
                                        bbox=self.plot_formatter.textbox_format,
                                        transform=self.ax.transAxes, ha="left"))
        # Additional, optional display items
        if self.display_eng_info:
            # Pendulum centroid
            sim_objects.append(AnimPoint(sprite_generator, sys.pendulum,
                                             sys.pendulum.get_centroid, self.ax))
            # Text boxes for state variables/input and disturbances
            sim_objects.append(AnimText(("x={x:.2f~P}\n"
                                            r"$v_x$={vx:.2f~P}" + "\n"
                                            r"$\theta$={theta:.2f~P}" + "\n"
                                            r"$\omega$={omega:.2f~P}" + "\n"
                                            r"u={u:.2f~P}"),
                                            self.ax, x=0.98, y=0.02,
                                            bbox=self.plot_formatter.textbox_format,
                                            transform=self.ax.transAxes, ha="right"))
            sim_objects.append(AnimText((r"$w_x$={w_x:.2f~P}" + "\n"
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

    def save(self, filename: str = "animation.gif") -> None:
        self.ani.save(filename, writer="pillow", fps=self.refresh_rate.magnitude)
