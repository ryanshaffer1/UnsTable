
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Self

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
    MINX_MINY_MAXZ = "minx_miny_maxz"
    MINX_MAXY_MINZ = "minx_maxy_minz"
    MINX_MAXY_MAXZ = "minx_maxy_maxz"
    MAXX_MINY_MINZ = "maxx_miny_minz"
    MAXX_MINY_MAXZ = "maxx_miny_maxz"
    MAXX_MAXY_MINZ = "maxx_maxy_minz"
    MAXX_MAXY_MAXZ = "maxx_maxy_maxz"

    @classmethod
    def from_string(cls, string: str) -> Self:
        for member in cls:
            if member.value == string:
                return member
        msg = f"Invalid string for BodyRefPoint: {string}"
        raise ValueError(msg)

    def offset_to_centroid(self) -> np.ndarray:
        point_type_to_offset = {
            BodyRefPoint.CENTER:            np.array((0, 0, 0)),
            BodyRefPoint.BOTTOM_CENTER:     np.array((0, 0, 0.5)),
            BodyRefPoint.TOP_CENTER:        np.array((0, 0, -0.5)),
            BodyRefPoint.MINX_MINY_MINZ:    np.array((0.5, 0.5, 0.5)),
            BodyRefPoint.MINX_MINY_MAXZ:    np.array((0.5, 0.5, -0.5)),
            BodyRefPoint.MINX_MAXY_MINZ:    np.array((0.5, -0.5, 0.5)),
            BodyRefPoint.MINX_MAXY_MAXZ:    np.array((0.5, -0.5, -0.5)),
            BodyRefPoint.MAXX_MINY_MINZ:    np.array((-0.5, 0.5, 0.5)),
            BodyRefPoint.MAXX_MINY_MAXZ:    np.array((-0.5, 0.5, -0.5)),
            BodyRefPoint.MAXX_MAXY_MINZ:    np.array((-0.5, -0.5, 0.5)),
            BodyRefPoint.MAXX_MAXY_MAXZ:    np.array((-0.5, -0.5, -0.5)),
        }
        return point_type_to_offset[self]

@dataclass(kw_only=True)
class RigidBody(ABC):
    name: str
    mass: Quantity
    origin_type: BodyRefPoint | str
    parent_frame: CoordFrame | None = None
    rotations: dict[str, str] | None = None
    body_frame: CoordFrame = field(default_factory=CoordFrame)
    origin_mount: list[Self, BodyRefPoint | str] | None = None
    _dimensions: np.ndarray = field(init=False)
    _parent_offset: np.ndarray | None = field(init=False)
    _rot_angle: Quantity = 0*ureg.degrees

    def __post_init__(self) -> None:
        # Clean up/process optional inputs
        if isinstance(self.origin_type, str):
            self.origin_type = BodyRefPoint.from_string(self.origin_type)
        if self.origin_mount:
            if isinstance(self.origin_mount[1], str):
                self.origin_mount[1] = BodyRefPoint.from_string(self.origin_mount[1])
            self.set_origin_from_other_body(*self.origin_mount)

        # Convert to base units
        self.mass.ito_base_units()
        self.set_parent_frame(self.parent_frame)

    # -------- MASS PROPERTIES --------

    def get_mass(self) -> Quantity:
        return self.mass

    def get_centroid(self) -> GlobalPoint:
        """Return the centroid as a GlobalPoint (absolute coordinates)."""
        offset_factors = self.origin_type.offset_to_centroid()
        offset = tuple(offset_factors * self._dimensions)
        return self.body_frame.to_global(Point(self.body_frame, *offset))

    @abstractmethod
    def get_moi_matrix(self, point: Point) -> np.ndarray:
        """Calculate the Moment of Inertia matrix about a pivot point. Implemented by subclasses."""

    def parallel_axis_theorem(self, point: Point, cg_moi: np.ndarray) -> np.ndarray:
        mass = self.get_mass()

        # Ensure both centroid and point are in global coordinates
        centroid_gp = self.get_centroid()
        point_gp = point.to_global() if isinstance(point, Point) else point

        # Offset from CG to pivot point
        offset = point_gp.vector_to(centroid_gp)
        offset_mag = offset.magnitude

        # Parallel axis term
        parallel_term = mass * (np.linalg.norm(offset_mag)**2 * np.identity(3) -
                                       np.outer(offset_mag, offset_mag)) * offset.units**2

        # Add parallel axis term to MOI about the CG
        return (cg_moi + parallel_term)

    # -------- COORDINATE FRAME MANIPULATIONS ---------

    def set_parent_frame(self, parent_frame: CoordFrame) -> None:
        if parent_frame is not None:
            self.parent_frame = parent_frame
            self._parent_offset = self.parent_frame.get_frame_offset(self.body_frame)

    def set_origin_from_other_body(self, other: Self, other_point_type: BodyRefPoint) -> None:
        origin_gp = other.get_point(other_point_type, cs_type="global")
        self.body_frame.set_init_origin(origin_gp)

    def update_frame(self, state: State) -> None:
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
        # Recursively assign the rot angle to all sub-bodies
        if hasattr(self, "bodies"):
            for body in self.bodies:
                body.hack_save_frame_rotation(angle)

    def get_frame_rotation(self, axis: str) -> Quantity:
        # Currently only support extracting rotation about Y from the DCM
        axis = axis.upper()
        if axis == "Y":
            return self._rot_angle
        return 0 * ureg.degree

    # -------- Rigid Body Point Manipulations --------

    def get_point(self, point_type: BodyRefPoint, cs_type: str = "body") -> Point | GlobalPoint:
        # Calculate body-frame offset distance components from centroid to desired point
        offset_factors = point_type.offset_to_centroid()
        offset = -offset_factors * self._dimensions
        # Calculate body-frame offset distance components from centroid to origin
        offset_to_origin_factors = self.origin_type.offset_to_centroid()
        # Add offsets to get distances from origin to desired point
        offset_to_origin = tuple(offset_to_origin_factors * self._dimensions + offset)
        # Create point in body frame with those distances
        local_point = Point(self.body_frame, *offset_to_origin)

        # Return the point in the desired CS
        return self.local_point_by_cs_type(local_point, cs_type)

    def local_point_by_cs_type(self, local_point: Point, cs_type: str) -> Point | GlobalPoint:
        # Do nothing if local/body frame is desired
        if cs_type == "body":
            return local_point

        # Convert the body-local point to a GlobalPoint
        if cs_type in ("global", "world"):
            return self.body_frame.to_global(local_point)

        # Throw error for unsupported CS type
        msg = f"Unsupported cs_type '{cs_type}' for get_point"
        raise ValueError(msg)

    def normalized_point_position(self, point_type: BodyRefPoint) -> np.ndarray:
        norm_dists_point_to_centroid = point_type.offset_to_centroid()
        norm_dists_origin_to_centroid = self.origin_type.offset_to_centroid()
        norm_dists_origin_to_point = (np.array(norm_dists_origin_to_centroid) -
                                      np.array(norm_dists_point_to_centroid))
        return self.body_frame.dcm @ norm_dists_origin_to_point

    def global_bounding_box(self, state: State) -> tuple[tuple[Quantity, Quantity]]:
        # Update the body based on the state
        self.update_frame(state)
        origin = self.body_frame.origin

        # Initialize min/max values for X, Y, and Z
        bounding_box = np.array(((np.nan, -np.nan), (np.nan, -np.nan), (np.nan, -np.nan)))

        # Get the normalized coordinates for the 8 corners of the body
        corner_types = [BodyRefPoint.MINX_MINY_MINZ, BodyRefPoint.MINX_MINY_MAXZ,
                        BodyRefPoint.MINX_MAXY_MINZ, BodyRefPoint.MINX_MAXY_MAXZ,
                        BodyRefPoint.MAXX_MINY_MINZ, BodyRefPoint.MAXX_MINY_MAXZ,
                        BodyRefPoint.MAXX_MAXY_MINZ, BodyRefPoint.MAXX_MAXY_MAXZ]
        for corner in corner_types:
            norm_dists = self.normalized_point_position(corner)
            # Keep track of the min and max normalized coordinates
            bounding_box = update_bounding_box(bounding_box, norm_dists)

        # Convert from normalized coordinates to global coordinates
        origin_array = origin.to_array_with_units()
        bounding_box = (bounding_box.T * self._dimensions + origin_array).T
        return bounding_box

@dataclass(kw_only=True)
class Block(RigidBody):

    width: Quantity # X dimension
    depth: Quantity # Y dimension
    height: Quantity # Z dimension
    name: str = "block"

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert to base units
        self.width.ito_base_units()
        self.depth.ito_base_units()
        self.height.ito_base_units()

        self._dimensions = array_with_units((self.width, self.depth, self.height))

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
    name: str = "cylinder"
    _dimensions: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert to base units
        self.length.ito_base_units()
        self.radius.ito_base_units()

        self._dimensions = array_with_units((2*self.radius, 2*self.radius, self.length))

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
    name: str = "sphere"
    _dimensions: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert to base units
        self.radius.ito_base_units()

        self._dimensions = array_with_units((2*self.radius, 2*self.radius, 2*self.radius))

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
    name: str = "body_system"
    mass: Quantity = field(init=False)

    def __post_init__(self) -> None:
        # Calculate mass from sub-bodies
        self.mass = self.get_mass()

        # Complete rigid body initialization to take care of some positioning/etc.
        super().__post_init__()

        # Set up coordinate frame hierarchy
        for body in self.bodies:
            body.set_parent_frame(self.body_frame)

        # Calculate mass properties which depend on sub-body positioning
        self.centroid = self.get_centroid()
        self.moi = self.get_moi_matrix(self.body_frame.origin)

    def get_mass(self) -> Quantity:
        return sum(body.get_mass() for body in self.bodies)

    def get_centroid(self) -> GlobalPoint:
        total_mass = self.get_mass()
        if total_mass == 0 * ureg.kg:
            return self.bodies[0].body_frame.origin
        # Calculate global x, y, and z coordinates via weighted sum of sub-body centroids
        x = sum(body.get_mass() * body.get_centroid().x for body in self.bodies) / total_mass
        y = sum(body.get_mass() * body.get_centroid().y for body in self.bodies) / total_mass
        z = sum(body.get_mass() * body.get_centroid().z for body in self.bodies) / total_mass
        # Return a global point of the centroid
        centroid_gp = GlobalPoint(x=x, y=y, z=z)
        return centroid_gp

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
            sub_body.update_frame(state)
        else:
            for body in self.bodies:
                body.update_frame(state)

    def global_bounding_box(self, state: State) -> tuple[tuple[Quantity, Quantity]]:
        # Update the rigid body system based on the state
        super().update_frame(state)

        # Initialize min/max values for X, Y, and Z
        bounding_box = np.array(((np.nan, -np.nan),
                                 (np.nan, -np.nan),
                                 (np.nan, -np.nan)))*ureg.meter

        # Calculate bounding box for each sub-body
        for body in self.bodies:
            sub_bbox = body.global_bounding_box(state)

            # Update overall bounding box using sub-body bounding box
            bounding_box = update_bounding_box(bounding_box, sub_bbox)

        return bounding_box

def array_with_units(tup: tuple[Quantity]) -> np.ndarray:
    units = tup[0].units
    unitless_vals = tuple(x.magnitude for x in tup)
    return np.array(unitless_vals) * units

@ureg.wraps(("=A"), ("=A","=A"), strict=False)
def update_bounding_box(bounding_box: np.ndarray,
                        new_points: np.ndarray) -> np.ndarray:
    if new_points.ndim == 1:
        new_points = new_points.reshape(-1,1)

    all_vals = np.concatenate((bounding_box, new_points), axis=1)
    min_vals = np.nanmin(all_vals, axis=1)
    max_vals = np.nanmax(all_vals, axis=1)
    bounding_box = np.stack((min_vals, max_vals)).T

    return bounding_box
