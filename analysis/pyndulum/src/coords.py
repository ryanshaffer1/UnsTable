from dataclasses import dataclass, field
from typing import Self

import numpy as np
from pint import Quantity

from src import ureg


def _to_base_magnitudes(x: Quantity, y: Quantity, z: Quantity) -> tuple[np.ndarray, Quantity]:
    """Return a numpy array of magnitudes in base units and the units used."""
    x.ito_base_units()
    y.ito_base_units()
    z.ito_base_units()
    units = x.units
    return np.array([x.magnitude, y.magnitude, z.magnitude]), units

def _from_magnitude_array(arr: np.ndarray, units: Quantity) -> tuple[Quantity]:
    """Return a tuple of quantities from the array and units to be applied."""
    return tuple(v * units for v in arr)


@dataclass
class GlobalPoint:
    """Absolute point in the global coordinate system (with units).

    Use this when you need an origin or a location expressed in the global
    frame. For local/frame-relative points, use `Point`.
    """
    x: Quantity = 0 * ureg.meter
    y: Quantity = 0 * ureg.meter
    z: Quantity = 0 * ureg.meter

    def add_offset(self, offset: tuple[Quantity, Quantity, Quantity]) -> Self:
        return GlobalPoint(x=self.x + offset[0], y=self.y + offset[1], z=self.z + offset[2])

    def to_array(self) -> tuple[np.ndarray, Quantity]:
        arr, units = _to_base_magnitudes(self.x, self.y, self.z)
        return arr, units

    def to_array_with_units(self) -> np.ndarray:
        arr, units = self.to_array()
        return arr * units

    def vector_to(self, other: Self) -> np.ndarray:
        arr, units = _to_base_magnitudes(other.x - self.x, other.y - self.y, other.z - self.z)
        return arr * units

    def copy(self) -> Self:
        return GlobalPoint(x=self.x, y=self.x, z=self.z)

@dataclass
class CoordFrame:
    """A coordinate frame defined by an origin (GlobalPoint) and a rotation
    matrix that maps frame vectors to global vectors:

        v_global = dcm @ v_frame

    Methods allow translating/rotating the frame and converting points
    between frames.
    """
    origin: GlobalPoint = field(default_factory=lambda: GlobalPoint(0 * ureg.meter,
                                                                    0 * ureg.meter,
                                                                    0 * ureg.meter))
    dcm: np.ndarray = field(default_factory=lambda: np.identity(3))
    init_origin: GlobalPoint = field(init=False)

    def __post_init__(self) -> None:
        self.init_origin = self.origin.copy()

    def translate(self,
                  dx: Quantity = 0 * ureg.meter,
                  dy: Quantity = 0 * ureg.meter,
                  dz: Quantity = 0 * ureg.meter) -> None:
        # Offset is in local coordinates - transform to global
        local_offset, units = _to_base_magnitudes(dx, dy, dz)
        global_offset = _from_magnitude_array(self.rotate_vector(local_offset), units)

        # Add global offset to origin
        self.origin = self.origin.add_offset(global_offset)

    def translate_to(self,
                     x: Quantity | None = None,
                     y: Quantity | None = None,
                     z: Quantity | None = None,
                     *args: list,
                     point: GlobalPoint | None = None) -> None:
        if point:
            self.origin = point
        else:
            if x is None:
                x = self.init_origin.x
            if y is None:
                y = self.init_origin.y
            if z is None:
                z = self.init_origin.z
            self.origin = GlobalPoint(x, y, z)

    def set_init_origin(self, origin: GlobalPoint) -> None:
        self.init_origin = origin.copy()
        self.reset_origin()

    def reset_origin(self) -> None:
        self.origin.x = self.init_origin.x
        self.origin.y = self.init_origin.y
        self.origin.z = self.init_origin.z

    def set_rotation(self, dcm: np.ndarray) -> None:
        self.dcm = np.array(dcm, dtype=float)

    def rotate_vector(self, vector: np.ndarray) -> np.ndarray:
        return self.dcm @ vector

    def align_to(self, other: Self) -> None:
        self.dcm = other.dcm.copy()

    def get_frame_offset(self, other: Self) -> np.ndarray:
        return self.origin.vector_to(other.origin)

    def to_global(self, point: "Point") -> GlobalPoint:
        """Convert a `Point` expressed in this frame to a `GlobalPoint`."""
        vec, units = _to_base_magnitudes(point.x, point.y, point.z)
        rotated = self.dcm @ vec
        origin_arr, _ = self.origin.to_array()
        result = rotated + origin_arr
        return GlobalPoint(result[0] * units, result[1] * units, result[2] * units)

    def from_global(self, gp: GlobalPoint) -> "Point":
        """Convert a `GlobalPoint` into this frame's coordinates (returns `Point`)."""
        g_arr, units = gp.to_array()
        local = np.linalg.inv(self.dcm) @ (g_arr - self.origin.to_array()[0])
        return Point(frame=self, x=local[0] * units, y=local[1] * units, z=local[2] * units)


@dataclass
class Point:
    """A point expressed in a particular `CoordFrame`.

    Use `to_global()` and `frame.to_global()` / `frame.from_global()` to convert
    to/from `GlobalPoint` representations.
    """
    frame: CoordFrame
    x: Quantity = 0 * ureg.meter
    y: Quantity = 0 * ureg.meter
    z: Quantity = 0 * ureg.meter

    def add_offset(self, offset: tuple[Quantity, Quantity, Quantity]) -> Self:
        return Point(frame=self.frame,
                     x=self.x + offset[0],
                     y=self.y + offset[1],
                     z=self.z + offset[2])

    def to_global(self) -> GlobalPoint:
        return self.frame.to_global(self)

    def as_array(self) -> tuple[np.ndarray, Quantity]:
        return _to_base_magnitudes(self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        return np.array((self.x, self.y, self.z))

    def vector_to(self, other: Self) -> np.ndarray:
        if self.frame is not other.frame:
            msg = "Vector between points in different frames: convert to a common frame first."
            raise ValueError(msg)
        arr, units = _to_base_magnitudes(other.x - self.x, other.y - self.y, other.z - self.z)
        return arr * units


def rotation_matrix(axis: str, angle: Quantity) -> np.ndarray:
    """Return a 3x3 rotation matrix about axis 'X','Y', or 'Z' by `angle` radians."""
    a = angle.to(ureg.radians).magnitude
    axis = axis.upper()
    if axis == "X":
        return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    if axis == "Y":
        return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    if axis == "Z":
        return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    msg = f"Unknown axis '{axis}' for rotation matrix"
    raise ValueError(msg)

# Default global origin and frame
global_origin = GlobalPoint(0 * ureg.meter, 0 * ureg.meter, 0 * ureg.meter)
global_frame = CoordFrame(origin=global_origin, dcm=np.identity(3))

