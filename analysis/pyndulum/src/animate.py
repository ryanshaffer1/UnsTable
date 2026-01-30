from abc import ABC, abstractmethod
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np

from src import ureg
from src.primitives import RectPrim, LinePrim, Process, State
from src.sim_components import System

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
    
    def update(self, state: State, time: float) -> Rectangle:
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
    
    def update(self, state: State, time: float) -> plt.Line2D:
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
    
    def update(self, state: State, time: float) -> plt.Text:
        text_vars = {**vars(state), "time": time}
        content = self.fmt.format(**text_vars)
        self.text.set_text(content)
        return self.text

class SimAnimator:
    def __init__(self, system: System, process: Process, frames: np.ndarray):
        self.sys = system
        self.process = process
        self.fig, self.ax = plt.subplots()
        self.format_plot()

        # Add animated objects and labels
        self.objects = [AnimRectangle(self.sys.cart, self.ax, color="gray"),
                        AnimLine(self.sys.pendulum, self.ax, lw=max(1,self.sys.pendulum.thickness.magnitude), color="brown"),
                        AnimText("Time: {time:.2f~P}", self.ax, x=0.02, y=0.02, bbox=self.textbox_format, transform=self.ax.transAxes, ha="left"),
                        AnimText("x={x:.2f~P}\n"+r"$v_x$={vx:.2f~P}"+"\n"+r"$\theta$={theta:.2f~P}"+"\n"+r"$\omega$={omega:.2f~P}",
                                 self.ax, x=0.98, y=0.02, bbox=self.textbox_format, transform=self.ax.transAxes, ha="right")
                        ]

        # Create animation object with callbacks to initialize and update methods
        self.ani = FuncAnimation(
            self.fig, 
            func=self.update, 
            frames=frames, 
            init_func=self.init_anim, 
            blit=True, # Blitting optimizes drawing
            interval = (frames[1] - frames[0]).to("millisecond").magnitude
        )

    def format_plot(self):
        # Format plot
        self.ax.xaxis.set_units(ureg.meter)
        self.ax.yaxis.set_units(ureg.meter)
        self.ax.set_xlim(-1*ureg.meter, 1*ureg.meter)
        self.ax.set_ylim(-1*ureg.meter, 1*ureg.meter)
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

    # Updates the process and graph
    def update(self, time: float) -> tuple[Rectangle, plt.Line2D, plt.Text, plt.Text]:
        self.process.update(time)
        state = self.process.get_state()

        # Updating the simulation objects
        artists = tuple(obj.update(state, time) for obj in self.objects)
        return artists

    def show(self):
        """Displays the animation."""
        plt.show()
