"""Blender-background checks for the motion importer camera cuts."""

from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np

BLENDER_TOOLS = Path(__file__).resolve().parents[1] / "tools" / "blender"
sys.path.insert(0, str(BLENDER_TOOLS))
import import_motion  # noqa: E402


class CinematicCameraTests(unittest.TestCase):
  def test_cycles_reports_and_honors_explicit_cpu_rendering(self):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    details = import_motion._configure_cycles(scene, bpy, "cpu")

    self.assertEqual(scene.render.engine, "CYCLES")
    self.assertEqual(scene.cycles.device, "CPU")
    self.assertEqual(details["requested"], "cpu")
    self.assertEqual(details["selected"], "CPU")
    self.assertEqual(details["gpu_devices"], [])

  def test_cycles_auto_mode_records_the_selected_device(self):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    details = import_motion._configure_cycles(scene, bpy, "auto")

    self.assertIn(details["selected"], {"CPU", "GPU"})
    self.assertEqual(scene.cycles.device, details["selected"])
    self.assertEqual(details["requested"], "auto")

  def test_range_suffixed_ffmpeg_output_is_renamed_to_requested_mp4(self):
    with tempfile.TemporaryDirectory(prefix="ascento_video_output_test_") as temp_dir:
      output = Path(temp_dir) / "render.mp4"
      ranged_output = Path(temp_dir) / "render0001-0144.mp4"
      ranged_output.write_bytes(b"encoded video")

      result = import_motion._finalize_video_output(output, 1, 144)

      self.assertEqual(result, output)
      self.assertEqual(output.read_bytes(), b"encoded video")
      self.assertFalse(ranged_output.exists())

  def test_front_camera_heading_bisects_wheel_contacts(self):
    from mathutils import Vector

    forward = Vector((1.0, 0.0, 0.0))
    left_point = Vector((0.45, 0.0, 0.0))
    right_point = Vector((-0.1, -0.35, 0.0))
    heading = import_motion._contact_bisector_heading(forward, left_point, right_point, Vector)
    baseline = Vector((left_point.x - right_point.x, left_point.y - right_point.y, 0.0))

    self.assertAlmostEqual(heading.dot(baseline), 0.0, delta=1.0e-6)
    self.assertGreater(heading.dot(forward), 0.0)

  def test_robot_materials_use_surface_specific_pbr_settings(self):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    cache = {}
    tire = import_motion._blender_material(bpy, cache, (0.10, 0.10, 0.10, 1.0), role="wheel")
    shell = import_motion._blender_material(bpy, cache, (0.80, 0.80, 0.80, 1.0), role="leg")
    lens = import_motion._blender_material(bpy, cache, (0.01, 0.01, 0.01, 1.0), role="head")
    tire_shader = tire.node_tree.nodes.get("Principled BSDF")
    shell_shader = shell.node_tree.nodes.get("Principled BSDF")
    lens_shader = lens.node_tree.nodes.get("Principled BSDF")

    self.assertGreater(tire_shader.inputs["Roughness"].default_value, 0.8)
    self.assertEqual(tire_shader.inputs["Metallic"].default_value, 0.0)
    self.assertLess(shell_shader.inputs["Metallic"].default_value, 0.1)
    self.assertGreater(shell_shader.inputs["Roughness"].default_value, 0.4)
    self.assertLess(shell_shader.inputs["Coat Weight"].default_value, 0.15)
    self.assertLess(lens_shader.inputs["Roughness"].default_value, 0.3)
    self.assertTrue(any(node.type == "BUMP" for node in tire.node_tree.nodes))

  def test_camera_shots_are_evenly_cut_and_follow_robot_motion(self):
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 13
    root_object = bpy.data.objects.new("Test_Root", None)
    scene.collection.objects.link(root_object)
    import_motion._configure_cycles(scene)
    self.assertEqual(scene.render.engine, "CYCLES")
    self.assertIsNone(bpy.data.objects.get("Ascento_Cinematic_Ground"))
    times = np.linspace(0.0, 0.5, 13)
    root_pos = np.column_stack((np.linspace(0.0, 3.0, 13), np.zeros(13), np.full(13, 0.6)))
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (13, 1))
    vertices = [
      (-0.3, -0.2, -1.0),
      (0.3, -0.2, -1.0),
      (0.3, 0.2, -1.0),
      (-0.3, 0.2, -1.0),
      (-0.3, -0.2, 0.2),
      (0.3, -0.2, 0.2),
      (0.3, 0.2, 0.2),
      (-0.3, 0.2, 0.2),
    ]
    mesh = bpy.data.meshes.new("Test_Robot_Mesh")
    mesh.from_pydata(vertices, [], [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2)])
    robot_mesh = bpy.data.objects.new("Ascento_test_visual_00", mesh)
    scene.collection.objects.link(robot_mesh)
    robot_mesh.parent = root_object
    robot_mesh["ascento_visual_mesh"] = True
    for frame, position, quaternion in zip(range(1, 14), root_pos, root_quat, strict=True):
      root_object.location = tuple(position)
      root_object.rotation_mode = "QUATERNION"
      root_object.rotation_quaternion = tuple(quaternion)
      root_object.keyframe_insert(data_path="location", frame=frame)
      root_object.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    import_motion._set_linear_animation([root_object])
    shot_names = ["low_front", "side_follow", "orbit"]

    segments = import_motion._camera_shot_ranges(1, 13, shot_names)
    self.assertEqual(
      segments,
      [("low_front", 1, 5), ("side_follow", 6, 9), ("orbit", 10, 13)],
    )

    cameras = import_motion._create_cinematic_cameras(
      bpy, scene, shot_names, times, root_pos, root_quat, 24.0, 1, 13
    )
    self.assertEqual(len(cameras), 3)
    markers = list(scene.timeline_markers)
    self.assertEqual(len(markers), 3)
    scene.frame_set(1)
    self.assertEqual(scene.camera, cameras[0][3])
    for index, (shot_name, start, end, camera) in enumerate(cameras, start=1):
      marker = next(item for item in markers if item.name == f"CUT_{index:02d}_{shot_name}")
      self.assertEqual(marker.frame, start)
      self.assertEqual(marker.camera, camera)
      self.assertFalse(camera.data.dof.use_dof)
      scene.frame_set(start)
      first_position = camera.location.copy()
      scene.frame_set(end)
      last_position = camera.location.copy()
      self.assertGreater((last_position - first_position).length, 0.01)
      scene.frame_set(start)
      depsgraph = bpy.context.evaluated_depsgraph_get()
      evaluated_mesh = robot_mesh.evaluated_get(depsgraph)
      corners = [evaluated_mesh.matrix_world @ Vector(point) for point in evaluated_mesh.bound_box]
      center = Vector(
        tuple((min(point[i] for point in corners) + max(point[i] for point in corners)) / 2 for i in range(3))
      )
      projected_center = world_to_camera_view(scene, camera, center)
      self.assertAlmostEqual(projected_center.x, 0.5, delta=0.025)
      self.assertAlmostEqual(projected_center.y, 0.5, delta=0.025)

    scene.frame_set(cameras[-1][1])
    self.assertEqual(scene.camera, cameras[-1][3])

  def test_depth_of_field_is_opt_in_and_tracks_the_focus_target(self):
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    times = np.asarray([0.0, 1.0])
    root_pos = np.asarray([[0.0, 0.0, 0.6], [1.0, 0.0, 0.6]])
    root_quat = np.tile(np.asarray([1.0, 0.0, 0.0, 0.0]), (2, 1))
    cuts = import_motion._create_cinematic_cameras(
      bpy, scene, ["low_front"], times, root_pos, root_quat, 1.0, 1, 2, depth_of_field=True
    )
    camera = cuts[0][3]
    self.assertTrue(camera.data.dof.use_dof)
    self.assertIsNotNone(camera.data.dof.focus_object)
    self.assertNotEqual(camera.data.dof.focus_object.name, "Ascento_Root")

  def test_warehouse_package_imports_real_mesh_and_lights_the_scene(self):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    with tempfile.TemporaryDirectory(prefix="ascento_warehouse_test_") as temp_dir:
      temp_root = Path(temp_dir)
      fbx_path = temp_root / "warehouse_test.fbx"
      bpy.ops.mesh.primitive_cube_add(size=2.0, location=(0.0, 0.0, 1.0))
      warehouse = bpy.context.object
      warehouse.name = "Warehouse_Test"
      warehouse.dimensions = (10.0, 8.0, 5.0)
      bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
      bpy.ops.export_scene.fbx(filepath=str(fbx_path), use_selection=True)
      archive_path = temp_root / "warehouse_test.zip"
      with zipfile.ZipFile(archive_path, "w") as archive:
        archive.write(fbx_path, "Warehouse/warehouse_test.fbx")

      bpy.ops.wm.read_factory_settings(use_empty=True)
      scene = bpy.context.scene
      import_motion._configure_cycles(scene)
      stage = import_motion._make_warehouse_stage(
        bpy,
        scene,
        archive_path,
        temp_root / "extracted",
        (4.0, -2.0, 0.5),
        6.0,
      )

    meshes = [obj for obj in scene.objects if obj.type == "MESH"]
    self.assertGreaterEqual(len(meshes), 1)
    self.assertEqual(stage["ascento_asset_name"], "Warehouse FBX Model Free")
    self.assertEqual(scene["ascento_warehouse_license"], "CC BY 4.0")
    self.assertIsNone(stage.parent)
    self.assertEqual(scene.render.engine, "CYCLES")
    self.assertEqual(
      sum(obj.type == "LIGHT" and obj.name.startswith("Ascento_Room_Light_") for obj in scene.objects),
      6,
    )
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = [
      obj.evaluated_get(depsgraph).matrix_world @ Vector(corner)
      for obj in meshes
      for corner in obj.evaluated_get(depsgraph).bound_box
    ]
    self.assertAlmostEqual(min(point.z for point in corners), 0.0, delta=1.0e-4)
    self.assertAlmostEqual(max(point.x for point in corners) - min(point.x for point in corners), 24.0, delta=0.01)
    self.assertFalse(scene.render.film_transparent)

  def test_wheel_floor_clearance_lifts_only_penetrating_frames(self):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 2
    root_object = bpy.data.objects.new("Ascento_Root", None)
    scene.collection.objects.link(root_object)
    root_object.location = (0.0, 0.0, 0.6)
    root_object.keyframe_insert(data_path="location", frame=1)
    root_object.location = (0.0, 0.0, 0.9)
    root_object.keyframe_insert(data_path="location", frame=2)
    import_motion._set_linear_animation([root_object])

    mesh = bpy.data.meshes.new("Test_Wheel_Mesh")
    mesh.from_pydata(
      [(-0.2, -0.2, -0.7), (0.2, -0.2, -0.7), (0.2, 0.2, -0.7), (-0.2, 0.2, -0.7),
       (-0.2, -0.2, -0.6), (0.2, -0.2, -0.6), (0.2, 0.2, -0.6), (-0.2, 0.2, -0.6)],
      [],
      [(0, 1, 2, 3), (4, 7, 6, 5)],
    )
    wheel = bpy.data.objects.new("Ascento_test_wheel_visual", mesh)
    scene.collection.objects.link(wheel)
    wheel.parent = root_object
    wheel["ascento_visual_mesh"] = True
    wheel["ascento_link_name"] = "ascento/wheel_left"

    clearance = import_motion._apply_wheel_floor_clearance(bpy, scene, root_object, 1, 2)
    self.assertIsNotNone(clearance)
    self.assertGreater(clearance.location.z, 0.0)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    evaluated_wheel = wheel.evaluated_get(depsgraph)
    first_bottom = min((evaluated_wheel.matrix_world @ Vector(corner)).z for corner in evaluated_wheel.bound_box)
    self.assertAlmostEqual(first_bottom, 0.0, delta=1.0e-4)
    scene.frame_set(2)
    bpy.context.view_layer.update()
    evaluated_wheel = wheel.evaluated_get(depsgraph)
    second_bottom = min((evaluated_wheel.matrix_world @ Vector(corner)).z for corner in evaluated_wheel.bound_box)
    self.assertAlmostEqual(second_bottom, 0.2, delta=1.0e-4)

  def test_grounded_wheels_are_levelled_and_both_touch_the_floor(self):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    root_object = bpy.data.objects.new("Ascento_Root", None)
    scene.collection.objects.link(root_object)
    root_object.location = (0.0, 0.0, 0.6)
    root_object.rotation_euler = (0.12, 0.0, 0.0)
    root_object.keyframe_insert(data_path="location", frame=1)
    root_object.keyframe_insert(data_path="rotation_euler", frame=1)
    import_motion._set_linear_animation([root_object])

    mesh = bpy.data.meshes.new("Test_Wheel_Mesh")
    mesh.from_pydata(
      [(-0.1, -0.05, -0.05), (0.1, -0.05, -0.05), (0.1, 0.05, -0.05), (-0.1, 0.05, -0.05),
       (-0.1, -0.05, 0.05), (0.1, -0.05, 0.05), (0.1, 0.05, 0.05), (-0.1, 0.05, 0.05)],
      [],
      [],
    )
    wheels = []
    for side, lateral, height in (("left", 0.2, -0.61), ("right", -0.2, -0.58)):
      wheel = bpy.data.objects.new(f"Ascento_{side}_wheel_visual", mesh)
      scene.collection.objects.link(wheel)
      wheel.parent = root_object
      wheel.location = (0.0, lateral, height)
      wheel["ascento_visual_mesh"] = True
      wheel["ascento_link_name"] = f"ascento/wheel_{side}"
      wheels.append(wheel)

    clearance = import_motion._apply_wheel_floor_clearance(
      bpy,
      scene,
      root_object,
      1,
      1,
      contacts=np.asarray([[1.0, 1.0]]),
      times=np.asarray([0.0]),
      fps=24.0,
    )
    self.assertIsNotNone(clearance)
    scene.frame_set(1)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bottoms = []
    for wheel in wheels:
      evaluated = wheel.evaluated_get(depsgraph)
      bottoms.append(min((evaluated.matrix_world @ Vector(corner)).z for corner in evaluated.bound_box))
    self.assertAlmostEqual(bottoms[0], 0.0, delta=1.0e-4)
    self.assertAlmostEqual(bottoms[1], 0.0, delta=1.0e-4)

  def test_camera_shot_plan_requires_a_frame_per_cut(self):
    with self.assertRaisesRegex(ValueError, "at least one frame per shot"):
      import_motion._camera_shot_ranges(1, 2, ["low_front", "side_follow", "orbit"])


if __name__ == "__main__":
  suite = unittest.defaultTestLoader.loadTestsFromTestCase(CinematicCameraTests)
  result = unittest.TextTestRunner(verbosity=2).run(suite)
  raise SystemExit(not result.wasSuccessful())
