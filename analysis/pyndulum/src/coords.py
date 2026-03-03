from dataclasses import dataclass, field
from typing import Self

import numpy as np
from pint import Quantity

from src import ureg


class GlobalPoint:
    def __init__(self,
                 x: Quantity=0*ureg.meter,
                 y: Quantity=0*ureg.meter,
                 z: Quantity=0*ureg.meter) -> None:
        self.x = x
        self.y = y
        self.z = z

        # Convert to base units
        self.x.ito_base_units()
        self.y.ito_base_units()
        self.z.ito_base_units()

    def add_offset(self, offset: tuple[Quantity, Quantity, Quantity]) -> Self:
        return GlobalPoint(x=self.x + offset[0],
                           y=self.y + offset[1],
                           z=self.z + offset[2])

    def vector_to(self, other: Self) -> np.ndarray:
        return np.array([(other.x - self.x).magnitude,
                         (other.y - self.y).magnitude,
                         (other.z - self.z).magnitude]) * self.x.units

    def vector_from(self, other: Self) -> np.ndarray:
        return np.array([(self.x - other.x).magnitude,
                         (self.y - other.y).magnitude,
                         (self.z - other.z).magnitude]) * self.x.units

# Instantiate a global origin point for use as a default reference
global_origin = GlobalPoint(0 * ureg.meter, 0 * ureg.meter, 0 * ureg.meter)

class GlobalCoordSystem:
    origin = global_origin

# Instantiate a global coordinate system for use as a default reference
global_coords = GlobalCoordSystem()

@dataclass
class CoordSystem:
    # All relative to "global" coordinate system
    origin: GlobalPoint = field(default_factory=GlobalPoint)
    rotations: tuple[tuple[str, str]] | None = None
    init_dcm: np.ndarray | None = None
    dcm: np.ndarray = field(init=False)
    rotation_angles: list[list[str, Quantity]] = field(init=False)

    def __post_init__(self) -> None:
        if self.init_dcm is None:
            self.init_dcm = np.identity(3)
        self.dcm = self.init_dcm
        self.set_rotations(self.rotations)

    def set_rotations(self, rotations: tuple[tuple[str, str]]) -> None:
        self.rotations = rotations
        if self.rotations is None:
            self.rotation_angles = None
            return
        self.rotation_angles = []
        for rot in self.rotations:
            axis, _ = rot
            self.rotation_angles.append([axis, 0]) # TODO for frames with initial rotationss

    def get_framed_point(self) -> "Point":
        return Point(frame=self,
                     x=self.origin.x,
                     y=self.origin.y,
                     z=self.origin.z)

    def set_origin_point(self, point: GlobalPoint) -> None:
        if isinstance(point, Point):
            self.origin = point.to_global()
        else:
            self.origin = point

    def translate(self,
                  x_dist: Quantity = 0*ureg.meter,
                  y_dist: Quantity = 0*ureg.meter,
                  z_dist: Quantity = 0*ureg.meter) -> None:
        self.origin = self.origin.add_offset((x_dist, y_dist, z_dist))

    def translate_to(self,
                     x_pos: Quantity | None = None,
                     y_pos: Quantity | None = None,
                     z_pos: Quantity | None = None) -> None:
        # Handle missing position inputs: maintain current value
        curr_origin = self.get_framed_point()
        if x_pos is None:
            x_pos = curr_origin.x
        if y_pos is None:
            y_pos = curr_origin.y
        if z_pos is None:
            z_pos = curr_origin.z

        self.set_origin_point(GlobalPoint(x_pos, y_pos, z_pos))

    def align(self, other: Self) -> None:
        self.dcm = other.dcm

    def rotate(self, angles: dict[str, float]) -> None:
        # Check if coordinate frame can be rotated
        if self.rotations is None:
            return
        # Iterate through elementary rotations (about CS axes)
        for rot, rot_angle in zip(self.rotations, self.rotation_angles, strict=True):
            # Get the axis to rotate about and the rotation angle (if provided)
            _, angle_name = rot
            angle = angles.get(angle_name, 0)
            # Add to the frame's rotation angles
            rot_angle[1] = angle

        # Build the rotation matrix for the combination rotation
        rotation_matrix = build_rotation_matrix(self.rotation_angles)
        # Apply the rotation matrix
        self.dcm = rotation_matrix @ self.dcm

    def rotate_to(self, angles: dict[str, float]) -> None:
        # First reset the DCM
        self.dcm = self.init_dcm
        # Then apply the rotation
        self.rotate(angles)

    def rotate_point_to(self,
                        point: GlobalPoint,
                        angles: dict[str, float],
                        pivot_point: GlobalPoint) -> GlobalPoint:
        self.rotate_to(angles)
        offset_from_pivot = point.vector_from(pivot_point)
        rotated_offset = self.dcm @ offset_from_pivot
        return pivot_point.add_offset(rotated_offset)

@dataclass
class Point:
    frame: CoordSystem
    x: Quantity = 0 * ureg.meter
    y: Quantity = 0 * ureg.meter
    z: Quantity = 0 * ureg.meter

    def __post_init__(self) -> None:
        # Convert to base units
        self.x.ito_base_units()
        self.y.ito_base_units()
        self.z.ito_base_units()

    def add_offset(self, offset: tuple[Quantity, Quantity, Quantity]) -> Self:
        return Point(frame=self.frame,
                   x=self.x + offset[0],
                   y=self.y + offset[1],
                   z=self.z + offset[2])

    def to_global(self) -> GlobalPoint:
        # WRONG
        return GlobalPoint(x = self.x, y = self.y, z = self.z)

    def vector_to(self, other: Self) -> np.ndarray:
        if self.frame != other.frame:
            msg = "Vector between two points in different frames not supported."
            raise ValueError(msg)
        return np.array([(other.x - self.x).magnitude,
                         (other.y - self.y).magnitude,
                         (other.z - self.z).magnitude]) * self.x.units

    def vector_from(self, other: Self) -> np.ndarray:
        if self.frame != other.frame:
            msg = "Vector between two points in different frames not supported."
            raise ValueError(msg)
        return np.array([(self.x - other.x).magnitude,
                         (self.y - other.y).magnitude,
                         (self.z - other.z).magnitude]) * self.x.units

def build_rotation_matrix(rotation_angles: list[list[str, float]]) -> np.ndarray:
    total_rot_matrix = np.identity(3)
    for rot in rotation_angles:
        axis, angle = rot

        match axis.upper():
            case "X":
                rotation_matrix = np.array([[1, 0, 0],
                                            [0, np.cos(angle), -np.sin(angle)],
                                            [0, np.sin(angle), np.cos(angle)]])
            case "Y":
                rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle)],
                                            [0, 1, 0],
                                            [-np.sin(angle), 0, np.cos(angle)]])
            case "Z":
                rotation_matrix = np.array([[np.cos(angle), -np.sin(angle), 0],
                                            [np.sin(angle), np.cos(angle), 0],
                                            [0, 0, 1]])

        total_rot_matrix = rotation_matrix @ total_rot_matrix
    return total_rot_matrix
