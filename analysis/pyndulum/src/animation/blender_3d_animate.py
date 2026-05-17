from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import bpy
import pandas as pd
from pint import Quantity
from tqdm import tqdm

from src import ureg
from src.system import Block, BodyRefPoint, Cylinder, RigidBody, RigidBodySystem, Sphere, System
from src.variables import State


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
                return self._create_block(source, format_spec)
            case Cylinder():
                return self._create_cylinder(source, format_spec)
            case Sphere():
                return self._create_sphere(source, format_spec)

    def _create_block(self, source: Block, format_spec: dict) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cube_add(size=1.0)
        obj = bpy.context.active_object
        obj.scale = (source.width.magnitude, source.depth.magnitude, source.height.magnitude)
        self._apply_material(obj, format_spec)
        return obj

    def _create_cylinder(self, source: Cylinder, format_spec: dict) -> bpy.types.Object:
        bpy.ops.mesh.primitive_cylinder_add(radius=source.radius.magnitude,
                                            depth=source.length.magnitude)
        obj = bpy.context.active_object
        self._apply_material(obj, format_spec)
        return obj

    def _create_sphere(self, source: Sphere, format_spec: dict) -> bpy.types.Object:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=source.radius.magnitude)
        obj = bpy.context.active_object
        self._apply_material(obj, format_spec)
        return obj

    def _apply_material(self, obj: bpy.types.Object, format_spec: dict) -> None:
        # Create or get material
        mat_name = format_spec.get("material", "default_material")
        mat = bpy.data.materials.get(mat_name)
        if not mat:
            mat = bpy.data.materials.new(name=mat_name)
            mat.diffuse_color = format_spec.get("color", (0.8, 0.8, 0.8, 1.0))
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

        # Get collection's format spec to apply to all bodies
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

class AnimText(AnimObject):
    def __init__(self, fmt: str, x: float, y: float, z: float = 0.0, **kwargs: dict) -> None:
        super().__init__(**kwargs)
        self.fmt = fmt
        bpy.ops.object.text_add()
        self.text_obj = bpy.context.active_object
        self.text_obj.location = (x, y, z)
        self.text_obj.data.align_x = "LEFT"
        self.text_obj.data.align_y = "BOTTOM"

    def initialize(self) -> bpy.types.Object:
        self.text_obj.data.body = ""
        return self.text_obj

    def update(self,
               state: State,
               time: Quantity,
               *args: tuple,
               **kwargs: dict,
               ) -> None:
        if not self.valid_time(time):
            self.text_obj.hide_viewport = True
            self.text_obj.hide_render = True
            return

        self.text_obj.hide_viewport = False
        self.text_obj.hide_render = False

        # Convert state values to display units
        display_state = state.to_display_units()


        text_vars = {**display_state, "time": time, **kwargs}
        content = self.fmt.format(**text_vars)
        self.text_obj.data.body = content


@dataclass
class BlenderSceneFormatter:
    scene_title: str = "Inverted Pendulum System 3D"
    camera_distance: float = 10.0
    camera_location: tuple[float, float, float] = (10, -10, 5)
    light_energy: float = 1000.0

    def setup_scene(self) -> None:
        # Set scene name
        bpy.context.scene.name = self.scene_title

        # Setup camera
        if not bpy.context.scene.camera:
            bpy.ops.object.camera_add()
            camera = bpy.context.active_object
            bpy.context.scene.camera = camera
        else:
            camera = bpy.context.scene.camera

        camera.location = self.camera_location
        camera.rotation_euler = (1.2, 0, 0.8)  # Look at origin roughly

        # Setup light
        bpy.ops.object.light_add(type="SUN")
        light = bpy.context.active_object
        light.location = (5, -5, 10)
        light.data.energy = self.light_energy

        # Set render settings for animation
        bpy.context.scene.render.fps = 30
        bpy.context.scene.frame_start = 0

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
        self.scene_formatter.setup_scene()

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
        # Text box for time
        sim_objects.append(AnimText("Time: {time:.2f~P}", x=-5, y=5, z=2))
        # Additional, optional display items
        if self.display_eng_info:
            # Pendulum centroid
            sim_objects.append(AnimPoint(obj_generator, sys.pendulum,
                                             sys.pendulum.get_centroid))
            # Text boxes for state variables/input and disturbances
            sim_objects.append(AnimText(("x={x:.2f~P}\n"
                                            "v_x={vx:.2f~P}" + "\n"
                                            "theta={theta:.2f~P}" + "\n"
                                            "omega={omega:.2f~P}" + "\n"
                                            "u={u:.2f~P}"),
                                            x=-5, y=4, z=2))
            sim_objects.append(AnimText(("w_x={w_x:.2f~P}" + "\n"
                                             "w_vx={w_vx:.2f~P}" + "\n"
                                             "w_theta={w_theta:.2f~P}" + "\n"
                                             "w_omega={w_omega:.2f~P}"),
                                             x=-5, y=3, z=2))

        # Set validity time interval for each object to control when they are displayed
        for sim_obj in sim_objects:
            sim_obj.times = sys.times

        return sim_objects

    def show(self) -> None:
        pass

    def save(self, filename: str = "animation.blend") -> None:
        bpy.ops.wm.save_as_mainfile(filepath=filename)
