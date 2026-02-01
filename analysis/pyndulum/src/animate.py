from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.primitives import RectPrim, LinePrim, State
from src.system import System


class AnimObject(ABC):
    @abstractmethod
    def initialize(self) -> plt.Artist:
        pass
    @abstractmethod
    def update(self, state: State, time: float) -> plt.Artist:
        pass

class AnimRectangle(AnimObject):
    def __init__(self, source: RectPrim, ax: plt.Axes, **kwargs):
        self.source = source
        self.rect = ax.add_patch(Rectangle((0,0), source.width, source.height, **kwargs))
    
    def initialize(self) -> Rectangle:
        self.rect.set_xy(([],[]))
        return self.rect
    
    def update(self, state: State, *args, **kwargs) -> Rectangle:
        ll_corner = self.source.get_ll_corner(state)
        self.rect.set_xy(coord for coord in ll_corner)
        return self.rect
    
class AnimLine(AnimObject):
    def __init__(self, source: LinePrim, ax: plt.Axes, **kwargs):
        self.source = source
        self.line, = ax.plot([], [], **kwargs)
    
    def initialize(self) -> plt.Line2D:
        self.line.set_data([], [])
        return self.line
    
    def update(self, state: State, *args, **kwargs) -> plt.Line2D:
        endpoints = self.source.get_endpoints(state)
        x_data = [endpoints[0][0], endpoints[1][0]]
        y_data = [endpoints[0][1], endpoints[1][1]]
        self.line.set_data(x_data, y_data)
        return self.line
    
class AnimText(AnimObject):
    def __init__(self, fmt: str, ax: plt.Axes, x: float, y: float, **kwargs):
        self.fmt = fmt
        self.text = ax.text(x, y, "", **kwargs)
    
    def initialize(self) -> plt.Text:
        self.text.set_text("")
        return self.text
    
    def update(self, state: State, time: Quantity, *args, **kwargs) -> plt.Text:
        # Convert state values to display units
        display_state = state.to_display_units()

        text_vars = {**display_state, "time": time, **kwargs}
        content = self.fmt.format(**text_vars)
        self.text.set_text(content)
        return self.text

class SimAnimator:
    def __init__(self,
                 system: System,
                 times: Quantity,
                 histories: dict[str, np.ndarray],
                 refresh_rate: Quantity = 30 * ureg.hertz,
                 ):
        self.sys = system
        self.state_history = histories["states"]
        self.input_history = histories["inputs"]
        self.times = times
        self.fig, self.ax = plt.subplots()
        self.refresh_rate = refresh_rate
        self.format_plot()

        # Add animated objects and labels
        self.objects = [AnimRectangle(self.sys.cart, self.ax, color="gray"),
                        AnimLine(self.sys.pendulum, self.ax, lw=max(1,self.sys.pendulum.thickness.magnitude), color="brown"),
                        AnimText("Time: {time:.2f~P}", self.ax, x=0.02, y=0.02, bbox=self.textbox_format, transform=self.ax.transAxes, ha="left"),
                        AnimText("x={x:.2f~P}\n"+r"$v_x$={vx:.2f~P}"+"\n"+r"$\theta$={theta:.2f~P}"+"\n"+r"$\omega$={omega:.2f~P}"+"\n"+r"u={u:.2f~P}",
                                 self.ax, x=0.98, y=0.02, bbox=self.textbox_format, transform=self.ax.transAxes, ha="right")
                        ]

        # Calculate frames to control animation refresh rate
        sim_time_step = times[1] - times[0]
        steps_per_frame = max(1, int(1 / self.refresh_rate / sim_time_step))
        frames = range(0, len(times), steps_per_frame)

        # Create animation object with callbacks to initialize and update methods
        self.ani = FuncAnimation(
            self.fig, 
            func=self.update, 
            frames=tqdm(frames), 
            init_func=self.init_anim, 
            blit=True, # Blitting optimizes drawing
            interval = int((1/self.refresh_rate).to("millisecond").magnitude)
        )

    def save(self, filename: str = "animation.gif"):
        self.ani.save(filename, writer="pillow", fps=self.refresh_rate.magnitude)

    def format_plot(self):
        # Calculate plot limits to keep cart/pendulum in view
        endpoint_history = self.sys.trace_pend_endpoint_history(self.state_history)
        xlims = (np.min(endpoint_history[[0,2],:]), np.max(endpoint_history[[0,2],:]))*ureg.meter
        ylims = (np.min(endpoint_history[[1,3],:]), np.max(endpoint_history[[1,3],:]))*ureg.meter
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
        self.ax.grid(True)
        
        # Set format for text boxes
        self.textbox_format = {'facecolor':'w', 'alpha':0.5, 'pad':5}

    def init_anim(self) -> tuple[Rectangle, plt.Line2D]:
        """Initializes the animation plot (runs once at the start)."""
        artists = tuple(obj.initialize() for obj in self.objects)
        # Must return an iterable of artists when blit=True
        return artists

    # Updates the graph
    def update(self, frame: float) -> tuple[Rectangle, plt.Line2D, plt.Text, plt.Text]:
        state = State.from_vector(self.state_history[:, frame])
        u = self.input_history[frame] * ureg.newton
        time = self.times[frame]

        # Updating the simulation objects
        artists = tuple(obj.update(state, time, u=u) for obj in self.objects)
        return artists

    def show(self):
        """Displays the animation."""
        plt.show()
