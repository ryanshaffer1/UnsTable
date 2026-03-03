from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Patch, Rectangle
from pint import Quantity

from src import ureg
from src.system import Block, BodyRefPoint, Cylinder, RigidBody, RigidBodySystem, Sphere
from src.variables import State


class AnimObject(ABC):
    @abstractmethod
    def initialize(self) -> plt.Artist:
        pass
    @abstractmethod
    def update(self, state: State, time: float) -> plt.Artist:
        pass

class AnimCollection:
    def __init__(self, source: RigidBodySystem, ax: plt.Axes, **kwargs: dict) -> None:
        self.patches = []
        self.source = source

        for body in source.bodies:
            match body:
                case Block():
                    self.patches.append(AnimRectangle(body, ax, collection=self, **kwargs))
                case Cylinder():
                    self.patches.append(AnimRectangle(body, ax, collection=self, **kwargs))
                case Sphere():
                    self.patches.append(AnimCircle(body, ax, collection=self, **kwargs))
                case _:
                    msg = f"Unsupported rigid body type for animation: {type(body)}"
                    raise NotImplementedError(msg)

    def update_object(self, obj: RigidBody, state: State) -> None:
        self.source.update_frame(state, obj)

class AnimRectangle(AnimObject):
    def __init__(self,
                 source: Block,
                 ax: plt.Axes,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        self.source = source
        rect = source.get_mpl_sprite(**kwargs)
        self.rect = ax.add_patch(rect)
        self.collection = collection

    def initialize(self) -> Rectangle:
        self.rect.set_xy(([],[]))
        return self.rect

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Rectangle:
        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        lll_corner = self.source.get_point(BodyRefPoint.MINX_MINY_MINZ, cs_type="global")
        rotation = self.source.get_frame_rotation("Y")
        self.rect.set_angle(-rotation.magnitude)
        self.rect.set_xy((lll_corner.x, lll_corner.z))
        return self.rect

class AnimCircle(AnimObject):
    def __init__(self,
                 source: Sphere,
                 ax: plt.Axes,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        self.source = source
        self.collection = collection
        circle = source.get_mpl_sprite(**kwargs)
        self.circle = ax.add_patch(circle)

    def initialize(self) -> Circle:
        self.circle.set_center(([],[]))
        return self.circle

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Circle:
        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        center = self.source.get_point(BodyRefPoint.CENTER, cs_type="global")
        self.circle.set_center((center.x, center.z))
        return self.circle


class AnimText(AnimObject):
    def __init__(self, fmt: str, ax: plt.Axes, x: float, y: float, **kwargs: dict) -> None:
        self.fmt = fmt
        self.text = ax.text(x, y, "", **kwargs)

    def initialize(self) -> plt.Text:
        self.text.set_text("")
        return self.text

    def update(self, state: State, time: Quantity, *args: tuple, **kwargs: dict) -> plt.Text:
        # Convert state values to display units
        display_state = state.to_display_units()

        text_vars = {**display_state, "time": time, **kwargs}
        content = self.fmt.format(**text_vars)
        self.text.set_text(content)
        return self.text

