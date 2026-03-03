
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pint import Quantity

from src import ureg
from src.coords import CoordSystem, Point
from src.variables import State


class BodyRefPoint(Enum):
    CENTER = "center"
    BOTTOM_CENTER = "bottom_center"
    TOP_CENTER = "top_center"
    MINX_MINY_MINZ = "minx_miny_minz"

    def offset_to_centroid(self) -> tuple[float, float, float]:
        match self:
            case BodyRefPoint.CENTER:
                return (0, 0, 0)
            case BodyRefPoint.BOTTOM_CENTER:
                return (0, 0, 0.5)
            case BodyRefPoint.TOP_CENTER:
                return (0, 0, -0.5)
            case BodyRefPoint.MINX_MINY_MINZ:
                return (0.5, 0.5, 0.5)
            case _:
                msg = f"Unsupported BlockRefPoint type {self} for centroid offset calculation"
                raise NotImplementedError(msg)

@dataclass(kw_only=True)
class RigidBody(ABC):
    mass: Quantity
    body_frame: CoordSystem
    origin_type: BodyRefPoint | None
    parent_frame: CoordSystem | None = None
    _dimensions: tuple = field(init=False)

    def __post_init__(self) -> None:
        # Convert to base units
        self.mass.ito_base_units()

    def get_mass(self) -> Quantity:
        return self.mass

    def get_point(self, point_type: BodyRefPoint, cs_type: str = "body") -> Point:
        # Calculate body-frame offset distance components from centroid to desired point
        offset_factors = point_type.offset_to_centroid()
        offset = tuple(-fac*dim for fac, dim in zip(offset_factors, self._dimensions, strict=True))
        # Calculate body-frame offset distance components from centroid to origin
        offset_to_origin_factors = self.origin_type.offset_to_centroid()
        # Add offsets to get distances from origin to desired point
        offset_to_origin = tuple(fac*dim + off for fac, dim, off in
                                    zip(offset_to_origin_factors,
                                        self._dimensions,
                                        offset, strict=True))
        if cs_type == "body":
            # Create point in body frame with those distances
            return Point(self.body_frame, *offset_to_origin)

        # To get global coordinates: rotate about pivot point, then add to the pivot's global coords
        offset_array = (np.array([off.magnitude for off in offset_to_origin]) * offset[0].units).T
        rotated_offset = self.body_frame.dcm @ offset_array
        return self.body_frame.origin.add_offset(rotated_offset)

    def get_centroid(self) -> Point:
        offset_factors = self.origin_type.offset_to_centroid()
        offset = tuple(fac*dim for fac, dim in zip(offset_factors, self._dimensions, strict=True))
        return self.body_frame.origin.add_offset(offset)

    def update_frame(self, state: State) -> None:
        self.body_frame.translate_to(x_pos=state.x)
        if self.parent_frame:
            # Align rotation with the parent frame, then add any applicable rotations from there
            self.body_frame.align(self.parent_frame)
            self.body_frame.rotate(angles={"theta": state.theta})
        else:
            # Rotate to the angles present in the state
            self.body_frame.rotate_to(angles={"theta": state.theta})

    def get_frame_rotation(self, axis: str) -> Quantity:
        if self.body_frame.rotation_angles is None:
            return 0 * ureg.degree
        for rot in self.body_frame.rotation_angles:
            rot_axis, rot_angle = rot
            if rot_axis == axis:
                return rot_angle
        return 0 * ureg.degree

    @abstractmethod
    def get_moi_matrix(self, point: Point) -> np.ndarray:
        pass

    def parallel_axis_theorem(self, point: Point, cg_moi: np.ndarray) -> np.ndarray:
        mass = self.get_mass()
        centroid = self.get_centroid()
        offset = np.array([(centroid.x - point.x).magnitude,
                           (centroid.y - point.y).magnitude,
                           (centroid.z - point.z).magnitude]) * centroid.x.units
        parallel_term = mass * (np.linalg.norm(offset)**2 * np.identity(3) -
                                np.outer(offset.magnitude, offset.magnitude)*offset.units**2)

        return cg_moi + parallel_term

@dataclass(kw_only=True)
class Block(RigidBody):

    width: Quantity # X dimension
    depth: Quantity # Y dimension
    height: Quantity # Z dimension

    def __post_init__(self) -> None:
        # Convert to base units
        self.width.ito_base_units()
        self.depth.ito_base_units()
        self.height.ito_base_units()

        self._dimensions = (self.width, self.depth, self.height)

    def get_volume(self) -> Quantity:
        return self.width * self.depth * self.height

    def get_bounding_box(self) -> tuple[Point, Point, Point, Point, Point, Point, Point, Point]:
        # Returns the 8 corners of the block as Points
        centroid = self.get_centroid()
        corners = []
        for dx in [-self.width / 2, self.width / 2]:
            for dy in [-self.depth / 2, self.depth / 2]:
                for dz in [-self.height / 2, self.height / 2]:
                    corner_offset = (dx, dy, dz)
                    corner_point = centroid.add_offset(corner_offset)
                    corners.append(corner_point)
        return tuple(corners)

    def get_moi_matrix(self, point: Point) -> np.ndarray:
        # For a block, the MOI matrix is diagonal in the local coordinate system
        mass = self.get_mass()
        moi_xx = (1/12) * mass * (self.height**2 + self.depth**2)
        moi_yy = (1/12) * mass * (self.width**2 + self.height**2)
        moi_zz = (1/12) * mass * (self.width**2 + self.depth**2)
        moi_cg = np.diag([moi_xx.magnitude, moi_yy.magnitude, moi_zz.magnitude])*moi_xx.units

        # Apply parallel axis theorem to shift MOI to the specified point
        return self.parallel_axis_theorem(point, moi_cg)

@dataclass(kw_only=True)
class Cylinder(RigidBody, ABC):
    length: Quantity
    radius: Quantity
    _dimensions: tuple = field(init=False)

    def __post_init__(self) -> None:
        # Convert to base units
        self.length.ito_base_units()
        self.radius.ito_base_units()

        self._dimensions = (2*self.radius, 2*self.radius, self.length)

    @abstractmethod
    def get_endpoints(self, state: State) -> tuple[tuple[float, float], tuple[float, float]]:
        pass

    def get_moi_matrix(self, point: Point) -> np.ndarray:
        # For a Cylinder, the MOI matrix is diagonal in the local coordinate system
        mass = self.get_mass()
        moi_xx = (1/4) * mass * self.radius**2 + (1/12) * mass * self.length**2
        moi_yy = (1/4) * mass * self.radius**2 + (1/12) * mass * self.length**2
        moi_zz = (1/2) * mass * self.radius**2
        moi_cg = np.diag([moi_xx.magnitude, moi_yy.magnitude, moi_zz.magnitude])*moi_xx.units

        # Apply parallel axis theorem to shift MOI to the specified point
        return self.parallel_axis_theorem(point, moi_cg)

@dataclass(kw_only=True)
class Sphere(RigidBody, ABC):
    radius: Quantity
    _dimensions: tuple = field(init=False)

    def __post_init__(self) -> None:
        # Convert to base units
        self.radius.ito_base_units()

        self._dimensions = (2*self.radius, 2*self.radius, 2*self.radius)

    def get_moi_matrix(self, point: Point) -> np.ndarray:
        # For a Cylinder, the MOI matrix is diagonal in the local coordinate system
        mass = self.get_mass()
        moi = (2/5) * mass * self.radius**2
        moi_cg = np.diag([moi.magnitude, moi.magnitude, moi.magnitude])*moi.units

        # Apply parallel axis theorem to shift MOI to the specified point
        return self.parallel_axis_theorem(point, moi_cg)

@dataclass(kw_only=True)
class RigidBodySystem(RigidBody):
    bodies: list[RigidBody]
    mass: Quantity = field(init=False)

    def __post_init__(self) -> None:
        if not all(body.body_frame == self.body_frame for body in self.bodies):
            msg = "Currently, all bodies must be defined in the same coordinate system"
            raise NotImplementedError(msg)

        self.mass = self.get_mass()

    def get_mass(self) -> Quantity:
        return sum(body.get_mass() for body in self.bodies)

    def get_centroid(self) -> Point:
        total_mass = self.get_mass()
        if total_mass == 0 * ureg.kg:
            return self.bodies[0].body_frame.origin
        x = sum(body.get_mass() * body.get_centroid().x for body in self.bodies) / total_mass
        y = sum(body.get_mass() * body.get_centroid().y for body in self.bodies) / total_mass
        z = sum(body.get_mass() * body.get_centroid().z for body in self.bodies) / total_mass
        return Point(frame=self.body_frame, x=x, y=y, z=z)

    def get_moi_matrix(self, point: Point) -> np.ndarray:
        moi_matrix = np.zeros((3, 3))
        for body in self.bodies:
            moi_matrix += body.get_moi_matrix(point)
        return moi_matrix

    def update_frame(self, state: State, update_body: RigidBody | None = None) -> None:
        # Apply transformations to the system CS
        # Translate to the position present in the state
        self.body_frame.translate_to(x_pos=state.x)
        # Rotate to the angles present in the state
        self.body_frame.rotate_to(angles={"theta": state.theta})

        # Apply transformations to the bodies
        if update_body:
            update_body.body_frame.translate_to(x_pos=state.x)
            origin = update_body.body_frame.origin
            transformed_origin = update_body.body_frame.rotate_point_to(origin,
                                                                 angles={"theta": state.theta},
                                                                 pivot_point=self.body_frame.origin)
            update_body.body_frame.set_origin_point(transformed_origin)
        else:
            for body in self.bodies:
                body.body_frame.translate_to(x_pos=state.x)
                origin = body.body_frame.origin
                transformed_origin = self.body_frame.rotate_point_to(origin,
                                                                    anges={"theta": state.theta},
                                                                    pivot_point=self.body_frame.origin)
                body.body_frame.set_origin_point(transformed_origin)

    def transform_body(body, translation, rotation, pivot_point):
        pass