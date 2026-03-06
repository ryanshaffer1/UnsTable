from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Self

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch, Rectangle
from pint import Quantity

from src import ureg
from src.system import Block, BodyRefPoint, Cylinder, RigidBody, RigidBodySystem, Sphere
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
        format_spec = format_spec if format_spec else {}
        format_spec.update(self.formatter.get_spec(source.name))

        match source:
            case Block():
                return Rectangle((0,0), source.width, source.height, **format_spec)
            case Cylinder():
                return Rectangle((0,0), 2*source.radius, source.length, **format_spec)
            case Sphere():
                return Circle((0,0), source.radius, **format_spec)

class AnimObject(ABC):
    @abstractmethod
    def initialize(self) -> plt.Artist:
        pass
    @abstractmethod
    def update(self, state: State, time: float) -> plt.Artist:
        pass

class AnimCollection:
    def __init__(self,
                 sprite_gen: SpriteGenerator,
                 source: RigidBodySystem,
                 ax: plt.Axes,
                 collection: Self | None = None,
                 **kwargs: dict) -> None:
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
                 collection: AnimCollection | None = None) -> None:
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        sprite = sprite_gen.get_sprite(self.source, format_spec=prev_format_spec)
        self.sprite = ax.add_patch(sprite)

    def initialize(self) -> Rectangle:
        self.sprite.set_xy(([],[]))
        return self.sprite

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Rectangle:
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
                 collection: AnimCollection | None = None) -> None:
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        sprite = sprite_gen.get_sprite(self.source, format_spec=prev_format_spec)
        self.sprite = ax.add_patch(sprite)

    def initialize(self) -> Circle:
        self.sprite.set_center(([],[]))
        return self.sprite

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Circle:
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

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Circle:
        self.source.update_frame(state)
        center = self.update_func()
        self.sprite.set_center((center.x, center.z))
        return self.sprite

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

