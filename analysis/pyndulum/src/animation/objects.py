from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from pint import Quantity

from src import ureg
from src.primitives import BeamPrim, BlockPrim, State


class AnimObject(ABC):
    @abstractmethod
    def initialize(self) -> plt.Artist:
        pass
    @abstractmethod
    def update(self, state: State, time: float) -> plt.Artist:
        pass

class AnimRectangle(AnimObject):
    def __init__(self, source: BlockPrim, ax: plt.Axes, **kwargs: dict) -> None:
        self.source = source
        self.rect = ax.add_patch(Rectangle((0,0), source.width, source.height, **kwargs))

    def initialize(self) -> Rectangle:
        self.rect.set_xy(([],[]))
        return self.rect

    def update(self, state: State, *args: tuple, **kwargs: dict) -> Rectangle:
        ll_corner = self.source.get_ll_corner(state)
        self.rect.set_xy(coord for coord in ll_corner)
        return self.rect

class AnimLine(AnimObject):
    def __init__(self, source: BeamPrim, ax: plt.Axes, **kwargs: dict) -> None:
        self.source = source
        self.line, = ax.plot([], [], **kwargs)

    def initialize(self) -> plt.Line2D:
        self.line.set_data([], [])
        return self.line

    def get_updated_pos(self, state: State) -> tuple[list, list]:
        endpoints = self.source.get_endpoints(state.x, state.theta)
        x_data = [endpoints[0][0], endpoints[1][0]]
        y_data = [endpoints[0][1], endpoints[1][1]]
        return x_data, y_data

    def update(self, state: State, *args: tuple, **kwargs: dict) -> plt.Line2D:
        self.line.set_data(*self.get_updated_pos(state))
        return self.line

class OffsetAnimLine(AnimLine):
    def __init__(self,
                 base_line: AnimLine,
                 ax: plt.Axes,
                 offset: tuple[str, Quantity, Quantity, Quantity],
                 length: Quantity,
                 **kwargs: dict,
                 ) -> None:
        super().__init__(source=None, ax=ax, **kwargs)
        self.base_line = base_line
        self.offset = offset
        self.length = length

    def update(self, state: State, *args: tuple, **kwargs: dict) -> plt.Line2D:
        base_x_data, base_y_data = self.base_line.get_updated_pos(state)
        base_line_angle = np.arctan2(base_y_data[1] - base_y_data[0],
                                     base_x_data[1] - base_x_data[0])
        base_pt = self.offset[0]
        match base_pt:
            case "start":
                base_pt = (base_x_data[0], base_y_data[0])
            case "center":
                base_pt = ((base_x_data[0] + base_x_data[1])/2, (base_y_data[0] + base_y_data[1])/2)
            case "end":
                base_pt = (base_x_data[1], base_y_data[1])
            case _:
                msg = f"Invalid offset reference point: {self.offset[0]}"
                raise ValueError(msg)

        offset_par = self.offset[1] # Base point to offset line center (parallel to base line)
        offset_perp = self.offset[2] # Base point to offset line center (perp to base line)
        offset_angle = self.offset[3] # Angle of offset line relative to base line

        line_center = self.point_offset_from_line(base_pt, base_line_angle, offset_par, offset_perp)
        line_angle = base_line_angle + offset_angle
        x_data, y_data = self.get_endpoints_from_center(line_center, line_angle, self.length)
        self.line.set_data(x_data, y_data)
        return self.line

    def get_endpoints_from_center(self,
                                  center_pt: tuple[Quantity, Quantity],
                                  line_angle: Quantity,
                                  length: Quantity,
                                  ) -> tuple[list[Quantity], list[Quantity]]:
        half_length = length / 2
        start_pt = self.point_offset_from_line(center_pt, line_angle, -half_length, 0*ureg.meter)
        end_pt = self.point_offset_from_line(center_pt, line_angle, half_length, 0*ureg.meter)
        return [start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]]

    def point_offset_from_line(self,
                               start_pt: tuple[Quantity, Quantity],
                               line_angle: Quantity,
                               offset_par: Quantity,
                               offset_perp: Quantity,
                               ) -> tuple[Quantity, Quantity]:
        offset_x = (offset_par * np.cos(line_angle) -
                    offset_perp * np.sin(line_angle))
        offset_y = (offset_par * np.sin(line_angle) +
                    offset_perp * np.cos(line_angle))
        return (start_pt[0] + offset_x, start_pt[1] + offset_y)

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

