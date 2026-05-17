from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import bpy
import numpy as np
import pandas as pd
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.system import Block, BodyRefPoint, Cylinder, RigidBody, RigidBodySystem, Sphere, System
from src.variables import State


def color_name_to_rgba(color_name: str) -> tuple[float, float, float, float]:
    """Convert CSS color name or hex to RGBA tuple."""
    colors = {
        "gray": (0.5, 0.5, 0.5, 1.0),
        "grey": (0.5, 0.5, 0.5, 1.0),
        "black": (0.0, 0.0, 0.0, 1.0),
        "white": (1.0, 1.0, 1.0, 1.0),
        "red": (1.0, 0.0, 0.0, 1.0),
        "green": (0.0, 1.0, 0.0, 1.0),
        "blue": (0.0, 0.0, 1.0, 1.0),
        "cyan": (0.0, 1.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0, 1.0),
        "yellow": (1.0, 1.0, 0.0, 1.0),
        "saddlebrown": (0.545, 0.271, 0.075, 1.0),
        "darkgreen": (0.0, 0.392, 0.0, 1.0),
        "orange": (1.0, 0.647, 0.0, 1.0),
        "purple": (0.627, 0.125, 0.941, 1.0),
        "brown": (0.647, 0.165, 0.165, 1.0),
        "lightgray": (0.827, 0.827, 0.827, 1.0),
        "lightgrey": (0.827, 0.827, 0.827, 1.0),
    }
    color_lower = color_name.lower().strip()
    if color_lower in colors:
        return colors[color_lower]
    return (0.8, 0.8, 0.8, 1.0)


class BlenderObjectFormatter:
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
            msg = "BlenderObjectFormatter provided invalid input."
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

class ObjectGenerator:
    def __init__(self, formatter: BlenderObjectFormatter) -> None:
        self.formatter = formatter

    def create_object(self, source: RigidBody, format_spec: dict | None = None) -> bpy.types.Object:
        # Set formatting
        format_spec = format_spec or {}
        format_spec.update(self.formatter.get_spec(source.name))

        match source:
            case Block():
                obj = self._create_block(source)
            case Cylinder():
                obj = self._create_cylinder(source)
            case Sphere():
                obj = self._create_sphere(source)
        self._apply_material(obj, format_spec)
        return obj

    def _create_block(self, source: Block) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.scale = (source.width.magnitude, source.depth.magnitude, source.height.magnitude)
        return obj

    def _create_cylinder(self, source: Cylinder) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cylinder_add(radius=source.radius.magnitude,
                                            depth=source.length.magnitude)
        obj = bpy.context.active_object
        return obj

    def _create_sphere(self, source: Sphere) -> bpy.types.Object:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=source.radius.magnitude)
        obj = bpy.context.active_object
        return obj

    def _apply_material(self, obj: bpy.types.Object, format_spec: dict) -> None:
        # Create or get material
        mat_name = format_spec.get("material", f"{obj.name}_Material")
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            # Try to get color from various format spec keys
            color_spec = format_spec.get("color") or format_spec.get("facecolor")
            if color_spec:
                color = (color_spec
                         if isinstance(color_spec, tuple)
                         else color_name_to_rgba(str(color_spec)))
            else:
                color = (0.8, 0.8, 0.8, 1.0)
            mat.diffuse_color = color
            mat.use_nodes = False
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

class AnimObject(ABC):
    def __init__(self, times: tuple[Quantity] | None = None, **kwargs: dict) -> None:
        self.times = times

    @abstractmethod
    def initialize(self) -> bpy.types.Object:
        pass
    @abstractmethod
    def update(self, state: State, time: Quantity, frame: int) -> None:
        pass

    def valid_time(self, time: Quantity) -> bool:
        return (self.times is None) or (self.times[0] <= time < self.times[1])

class AnimCollection:
    def __init__(self,
                 obj_gen: ObjectGenerator,
                 source: RigidBodySystem,
                 collection: Self | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.objects = []
        self.source = source
        self.collection = collection

        # Get collection"s format spec to apply to all bodies
        self.format_spec = collection.format_spec if collection else {}
        self.format_spec.update(obj_gen.formatter.get_spec(self.source.name))

        for body in source.bodies:
            match body:
                case Block():
                    self.objects.append(AnimBlock(obj_gen, body, collection=self))
                case Cylinder():
                    self.objects.append(AnimCylinder(obj_gen, body, collection=self))
                case Sphere():
                    self.objects.append(AnimSphere(obj_gen, body, collection=self))
                case RigidBodySystem():
                    subsystem = AnimCollection(obj_gen, body, collection=self)
                    self.objects += subsystem.objects
                case _:
                    msg = f"Unsupported rigid body type for animation: {type(body)}"
                    raise NotImplementedError(msg)

    def update_object(self, obj: RigidBody, state: State) -> None:
        self.source.update_frame(state, obj)

class AnimBlock(AnimObject):
    def __init__(self,
                 obj_gen: ObjectGenerator,
                 source: Block,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        self.obj = obj_gen.create_object(self.source, format_spec=prev_format_spec)

    def initialize(self) -> bpy.types.Object:
        return self.obj

    def update(self,
               state: State,
               time: Quantity,
               frame: int,
               *args: tuple,
               **kwargs: dict,
               ) -> None:
        if not self.valid_time(time):
            self.obj.hide_viewport = True
            self.obj.hide_render = True
            return
        self.obj.hide_viewport = False
        self.obj.hide_render = False

        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        center_pt = self.source.get_point(BodyRefPoint.CENTER, cs_type="global")
        rotation = self.source.get_frame_rotation("Y")
        self.obj.location = (center_pt.x.magnitude, center_pt.y.magnitude, center_pt.z.magnitude)
        self.obj.rotation_euler = (0, rotation.to("radians").magnitude, 0)

        # Set keyframes
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)

class AnimCylinder(AnimObject):
    def __init__(self,
                 obj_gen: ObjectGenerator,
                 source: Cylinder,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        self.obj = obj_gen.create_object(self.source, format_spec=prev_format_spec)

    def initialize(self) -> bpy.types.Object:
        return self.obj

    def update(self,
               state: State, time: Quantity, frame: int, *args: tuple, **kwargs: dict) -> None:
        if not self.valid_time(time):
            self.obj.hide_viewport = True
            self.obj.hide_render = True
            return
        self.obj.hide_viewport = False
        self.obj.hide_render = False

        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        center_pt = self.source.get_point(BodyRefPoint.CENTER, cs_type="global")
        rotation = self.source.get_frame_rotation("Y")
        self.obj.location = (center_pt.x.magnitude, center_pt.y.magnitude, center_pt.z.magnitude)
        self.obj.rotation_euler = (0, rotation.to("radians").magnitude, 0)

        # Set keyframes
        self.obj.keyframe_insert(data_path="location", frame=frame)
        self.obj.keyframe_insert(data_path="rotation_euler", frame=frame)

class AnimSphere(AnimObject):
    def __init__(self,
                 obj_gen: ObjectGenerator,
                 source: Sphere,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.source = source
        self.collection = collection
        prev_format_spec = collection.format_spec if collection else None
        self.obj = obj_gen.create_object(self.source, format_spec=prev_format_spec)

    def initialize(self) -> bpy.types.Object:
        return self.obj

    def update(self,
               state: State,
               time: Quantity,
               frame: int,
               *args: tuple,
               **kwargs: dict,
               ) -> None:
        if not self.valid_time(time):
            self.obj.hide_viewport = True
            self.obj.hide_render = True
            return
        self.obj.hide_viewport = False
        self.obj.hide_render = False

        if self.collection:
            self.collection.update_object(self.source, state)
        else:
            self.source.update_frame(state)

        center = self.source.get_point(BodyRefPoint.CENTER, cs_type="global")
        self.obj.location = (center.x.magnitude, center.y.magnitude, center.z.magnitude)

        # Set keyframes
        self.obj.keyframe_insert(data_path="location", frame=frame)

class AnimPoint(AnimObject):
    def __init__(self,
                 obj_gen: ObjectGenerator,
                 source: RigidBody,
                 update_func: Callable,
                 collection: AnimCollection | None = None,
                 **kwargs: dict) -> None:
        super().__init__(**kwargs)
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
        self.obj = obj_gen.create_object(point_sphere, format_spec=prev_format_spec)

    def initialize(self) -> bpy.types.Object:
        return self.obj

    def update(self,
               state: State,
               time: Quantity,
               frame: int, *args: tuple, **kwargs: dict) -> None:
        if not self.valid_time(time):
            self.obj.hide_viewport = True
            self.obj.hide_render = True
            return
        self.obj.hide_viewport = False
        self.obj.hide_render = False

        self.source.update_frame(state)
        center = self.update_func()
        self.obj.location = (center.x.magnitude, center.y.magnitude, center.z.magnitude)

        # Set keyframes
        self.obj.keyframe_insert(data_path="location", frame=frame)


@dataclass
class BlenderSceneFormatter:
    scene_title: str = "Inverted Pendulum System 3D"
    camera_distance: float = 10.0
    camera_location: tuple[float, float, float] = (10, -10, 5)
    light_energy: float = 1000.0
    limits: tuple[tuple[Quantity, Quantity], tuple[Quantity, Quantity]] | None = None

    def setup_scene(self, systems: list[System], history: pd.DataFrame) -> None:
        # Delete default cube if it exists
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name == "Cube":
                bpy.data.objects.remove(obj, do_unlink=True)

        # Set scene name
        bpy.context.scene.name = self.scene_title

        # Setup camera
        if not bpy.context.scene.camera:
            bpy.ops.object.camera_add()
            camera = bpy.context.active_object
            bpy.context.scene.camera = camera
        else:
            camera = bpy.context.scene.camera

        # Calculate camera position based on limits if provided
        self._setup_camera_from_limits(camera, systems, history)
        bpy.context.scene.camera = camera

        # Setup light
        bpy.ops.object.light_add(type="SUN")
        light = bpy.context.active_object
        light.location = (5, -5, 10)
        light.data.energy = self.light_energy

        # Set render settings for animation
        bpy.context.scene.render.fps = 30
        bpy.context.scene.frame_start = 0
        bpy.context.scene.render.resolution_x = 1920
        bpy.context.scene.render.resolution_y = 1080

    def get_scene_limits(self,
                         systems: list[System],
                         history: pd.DataFrame,
                         ) -> list[list[Quantity]]:
        # Initialize limits to be updated based on system bounding boxes
        limits = [[0*ureg.meter,0*ureg.meter], [0*ureg.meter,0*ureg.meter]]

        for sys in systems:
            # Calculate the plot limits to always keep the system in view
            bbox_x, _, bbox_z = sys.get_bounding_box(history)
            xlims = bbox_x
            ylims = bbox_z
            # Add margin and make sure the origin is in view
            margin = 0.5 * ureg.meter
            xlims = [min(0, xlims[0]) - margin, max(0, xlims[1]) + margin]
            ylims = [min(0, ylims[0]) - margin, max(0, ylims[1]) + margin]
            # Update based on previously computed limits
            limits = [[min(xlims[0], limits[0][0]), max(xlims[1], limits[0][1])]
                      ,[min(ylims[0], limits[1][0]), max(ylims[1], limits[1][1])]]
        return limits


    def _setup_camera_from_limits(self,
                                  camera: bpy.types.Object,
                                  systems: list[System],
                                  history: pd.DataFrame,
                                  ) -> None:
        """Setup camera position and zoom based on system limits."""
        limits = self.limits or self.get_scene_limits(systems, history)
        x_limits = limits[0]
        z_limits = limits[1]

        # Extract magnitude values
        x_min = x_limits[0].magnitude if hasattr(x_limits[0], "magnitude") else float(x_limits[0])
        x_max = x_limits[1].magnitude if hasattr(x_limits[1], "magnitude") else float(x_limits[1])
        z_min = z_limits[0].magnitude if hasattr(z_limits[0], "magnitude") else float(z_limits[0])
        z_max = z_limits[1].magnitude if hasattr(z_limits[1], "magnitude") else float(z_limits[1])

        # Calculate center and extents
        center_x = (x_min + x_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        extent_x = abs(x_max - x_min) / 2.0
        extent_z = abs(z_max - z_min) / 2.0
        max_extent = max(extent_x, extent_z)

        # Position camera to view the scene
        distance = max_extent * 1.5
        camera.location = (center_x, -distance, center_z+distance/4)

        # Point camera at center using direction vector
        camera_pos = np.array(camera.location)
        target_pos = np.array([center_x, 0, center_z])
        direction = target_pos - camera_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
            dx, dy, dz = direction
            # Align Blender camera's local -Z axis with the desired direction.
            rx = np.arccos(np.clip(-dz, -1.0, 1.0))
            rz = np.arctan2(-dx, dy)
            camera.rotation_mode = "XYZ"
            camera.rotation_euler = (rx, 0.0, rz)

        # Set orthographic view with appropriate scale
        camera.data.type = "ORTHO"
        camera.data.ortho_scale = max_extent * 2

class Blender3dAnimator:
    def __init__(self,
                 scene_formatter: BlenderSceneFormatter | None = None,
                 obj_formatter: BlenderObjectFormatter | None = None,
                 refresh_rate: Quantity = 30 * ureg.hertz,
                 *args: list,
                 display_eng_info: bool = False) -> None:
        self.scene_formatter = scene_formatter or BlenderSceneFormatter()
        self.obj_formatter = obj_formatter or BlenderObjectFormatter()
        self.refresh_rate = refresh_rate
        self.display_eng_info = display_eng_info

    def create_system_animation(self,
                                systems: list[System],
                                times: Quantity,
                                history_df: pd.DataFrame,
                                *args: list,
                                show_progress: bool = True,
                                ) -> None:
        # Assign attributes for ease of access in update func
        self.systems = systems
        self.times = times
        self.history = history_df

        # Setup scene
        self.scene_formatter.setup_scene(systems, history_df)

        # Create animation objects
        self.objects = []
        for sys in self.systems:
            self.objects.extend(self.gen_anim_objects(sys))

        # Calculate frames to control animation refresh rate
        sim_time_step = times[1] - times[0]
        steps_per_frame = max(1, int(1 / self.refresh_rate / sim_time_step))
        frames = range(0, len(times), steps_per_frame)

        # Set end frame
        bpy.context.scene.frame_end = len(frames) - 1

        # Wrap times in tqdm to show progress bar
        if show_progress:
            frames = tqdm(frames)

        # Animate by setting keyframes at each frame
        for frame_idx, time_idx in enumerate(frames):
            time = self.times[time_idx]
            sim_info = self.history.loc[time].to_dict()
            state_dict = {var: sim_info.pop(var) for var in State.get_variable_names()}
            state = State(**state_dict)

            # Set current frame
            bpy.context.scene.frame_set(frame_idx)

            # Updating the simulation objects
            for obj in self.objects:
                obj.update(state, time, frame_idx, **sim_info)

    def gen_anim_objects(self, sys: System) -> list[AnimObject]:
        obj_generator = ObjectGenerator(self.obj_formatter)
        sim_objects: list[AnimObject] = []
        # Cart and pendulum
        sim_objects.append(AnimBlock(obj_generator, sys.cart))
        sim_objects += AnimCollection(obj_generator, sys.pendulum).objects
        # (text display removed)
        # Additional, optional display items
        if self.display_eng_info:
            # Pendulum centroid
            sim_objects.append(AnimPoint(obj_generator, sys.pendulum,
                                             sys.pendulum.get_centroid))
            # (text display removed)

        # Set validity time interval for each object to control when they are displayed
        for sim_obj in sim_objects:
            sim_obj.times = sys.times

        return sim_objects

    def show(self) -> None:
        pass

    def save(self, filename: str = "animation.blend") -> None:
        bpy.ops.wm.save_as_mainfile(filepath=filename)
