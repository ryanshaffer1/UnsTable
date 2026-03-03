
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from pint import Quantity

from src import ureg
from src.coords import CoordFrame, GlobalPoint, Point, rotation_matrix
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
    body_frame: CoordFrame
    origin_type: BodyRefPoint | None
    parent_frame: CoordFrame | None = None
    rotations: dict[str, str] | None = None
    _dimensions: tuple = field(init=False)
    _parent_offset: np.ndarray | None = field(init=False)
    _rot_angle: Quantity = 0*ureg.degrees

    def __post_init__(self) -> None:
        # Convert to base units
        self.mass.ito_base_units()
        self.set_parent_frame(self.parent_frame)

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
        # Create point in body frame with those distances
        local_point = Point(self.body_frame, *offset_to_origin)

        if cs_type == "body":
            return local_point

        if cs_type in ("global", "world"):
            # Convert the body-local point to a GlobalPoint
            return self.body_frame.to_global(local_point)
        msg = f"Unsupported cs_type '{cs_type}' for get_point"
        raise ValueError(msg)

    def get_centroid(self) -> GlobalPoint:
        """Return the centroid as a GlobalPoint (absolute coordinates)."""
        offset_factors = self.origin_type.offset_to_centroid()
        offset = tuple(fac * dim for fac, dim in zip(offset_factors, self._dimensions, strict=True))
        return self.body_frame.to_global(Point(self.body_frame, *offset))

    def set_parent_frame(self, parent_frame: CoordFrame) -> None:
        if parent_frame is not None:
            self.parent_frame = parent_frame
            self._parent_offset = self.parent_frame.get_frame_offset(self.body_frame)

    def update_frame(self, state: State, pivot_point: GlobalPoint | None = None) -> None:
        if self.parent_frame:
            # Set origin to parent origin
            self.body_frame.translate_to(point=self.parent_frame.origin)
            # Align frame rotations
            self.body_frame.align_to(self.parent_frame)
            # Apply offset to get new origin
            self.body_frame.translate(*self._parent_offset)

        else:
            # Translate frame origin to state's x (preserve other coordinates)
            self.body_frame.translate_to(x=state.x)

            # Set rotation angle about Y if the body rotates based on state
            if self.rotations and "Y" in self.rotations:
                angle = getattr(state, self.rotations["Y"])
                self.hack_save_frame_rotation(angle)
                self.body_frame.set_rotation(rotation_matrix("Y", angle))

    def hack_save_frame_rotation(self, angle: Quantity) -> None:
        # TODO: eventually want to make _rot_angle obsolete by
        # extracting this information from the frame's DCM.
        # Currently in a simplified 2D case, the DCM can only recreate
        # angles up to +/- 90 degrees.
        self._rot_angle = angle
        # Assign the rot angle to all sub-bodies
        if hasattr(self, "bodies"):
            for body in self.bodies:
                body._rot_angle = angle

    def get_frame_rotation(self, axis: str) -> Quantity:
        # Currently only support extracting rotation about Y from the DCM
        axis = axis.upper()
        if axis == "Y":
            return self._rot_angle
        return 0 * ureg.degree

    @abstractmethod
    def get_moi_matrix(self, point: Point) -> np.ndarray:
        pass

    def parallel_axis_theorem(self, point: Point, cg_moi: np.ndarray) -> np.ndarray:
        mass = self.get_mass()

        # Ensure both centroid and point are in global coordinates
        centroid_gp = self.get_centroid()
        point_gp = point.to_global() if isinstance(point, Point) else point

        c_arr, units = centroid_gp.to_array()
        p_arr, _ = point_gp.to_array()

        offset = c_arr - p_arr
        mass_val = mass.to_base_units().magnitude

        # numeric parallel term (unitless magnitudes)
        parallel_numeric = mass_val * (np.linalg.norm(offset)**2 * np.identity(3) -
                                       np.outer(offset, offset))* cg_moi.units

        # Add to MOI about the CG
        return (cg_moi + parallel_numeric)

@dataclass(kw_only=True)
class Block(RigidBody):

    width: Quantity # X dimension
    depth: Quantity # Y dimension
    height: Quantity # Z dimension

    def __post_init__(self) -> None:
        super().__post_init__()

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
        super().__post_init__()

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
        super().__post_init__()

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
        super().__post_init__()
        if not all(body.parent_frame == self.body_frame for body in self.bodies):
            msg = "All bodies in a RigidBodySystem must have matching parent frames."
            raise ValueError(msg)

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

    def update_frame(self, state: State, sub_body: RigidBody | None = None) -> None:
        # Apply transformations to the system CS
        super().update_frame(state)

        # Apply transformations to the bodies
        if sub_body:
            sub_body.update_frame(state, pivot_point=self.body_frame.origin)
        else:
            for body in self.bodies:
                body.update_frame(state, pivot_point=self.body_frame.origin)
