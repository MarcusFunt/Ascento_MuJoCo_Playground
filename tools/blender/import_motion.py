"""Create a Blender animation scene from an Ascento motion capture and URDF.

Run with Blender's bundled Python:

  blender --background --python tools/blender/import_motion.py -- \
    --capture captures/jump/take_000_clip.npz \
    --description ascento_description.zip \
    --output captures/blender/take_000.blend
"""

from __future__ import annotations

import argparse
import json
import math
import stat
import struct
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


SIM_TO_URDF_JOINT = {
  "left_hip": "ascento/hip_left",
  "left_knee": "ascento/knee_left",
  "left_wheel_joint": "ascento/ankle_left",
  "right_hip": "ascento/hip_right",
  "right_knee": "ascento/knee_right",
  "right_wheel_joint": "ascento/ankle_right",
}
DEFAULT_CAPTURE_JOINT_NAMES = tuple(SIM_TO_URDF_JOINT)
DEFAULT_ROOT_LINK = "ascento/base"
DEFAULT_ROBOT_COLOR = (0.67, 0.70, 0.74, 1.0)
CINEMATIC_SHOTS = ("low_front", "side_follow", "rear_chase", "orbit")
DEFAULT_WAREHOUSE_PATH = Path(__file__).resolve().parent / "assets" / "warehouse_fbx_model_free.zip"


def _tag(element: ET.Element) -> str:
  return element.tag.rsplit("}", 1)[-1]


def _child(element: ET.Element, name: str) -> ET.Element | None:
  return next((item for item in element if _tag(item) == name), None)


def _children(element: ET.Element, name: str) -> list[ET.Element]:
  return [item for item in element if _tag(item) == name]


def _parse_numbers(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
  if not value:
    return default
  values = tuple(float(part) for part in value.split())
  if len(values) != len(default):
    raise ValueError(f"expected {len(default)} numeric values, got {len(values)}")
  return values


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
  base = destination.resolve()
  with zipfile.ZipFile(archive_path) as archive:
    for info in archive.infolist():
      member = PurePosixPath(info.filename)
      if member.is_absolute() or ".." in member.parts or (member.parts and ":" in member.parts[0]):
        raise ValueError(f"unsafe path in package archive: {info.filename}")
      target = (base / Path(*member.parts)).resolve()
      try:
        target.relative_to(base)
      except ValueError as error:
        raise ValueError(f"unsafe path in package archive: {info.filename}") from error
      mode = info.external_attr >> 16
      if stat.S_ISLNK(mode):
        raise ValueError(f"symbolic links are not supported in package archives: {info.filename}")
      if info.is_dir():
        target.mkdir(parents=True, exist_ok=True)
      else:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))


def _description_paths(description: Path, temporary_root: Path) -> tuple[Path, Path]:
  source = description.expanduser().resolve()
  if not source.exists():
    raise FileNotFoundError(f"robot description does not exist: {source}")
  if source.suffix.lower() == ".zip":
    _safe_extract_zip(source, temporary_root)
    search_root = temporary_root
  else:
    search_root = source

  if search_root.is_file() and search_root.suffix.lower() == ".urdf":
    urdf_path = search_root
  elif search_root.is_dir():
    candidates = sorted(search_root.rglob("*.urdf"))
    if not candidates:
      raise FileNotFoundError(f"no URDF file found in {search_root}")
    preferred = [path for path in candidates if path.name.lower() == "ascento.urdf"]
    urdf_path = (preferred or candidates)[0]
    if len(candidates) > 1 and not preferred:
      raise ValueError(f"multiple URDF files found in {search_root}; pass the URDF file directly")
  else:
    raise ValueError("--description must be a URDF file, package directory, or ZIP archive")

  package_root = next(
    (parent for parent in (urdf_path.parent, *urdf_path.parents) if (parent / "package.xml").is_file()),
    urdf_path.parent.parent,
  )
  return urdf_path, package_root


def _package_name(package_root: Path, robot: ET.Element) -> str:
  manifest = package_root / "package.xml"
  if manifest.is_file():
    root = ET.parse(manifest).getroot()
    name = _child(root, "name")
    if name is not None and name.text:
      return name.text.strip()
  return robot.get("name", "ascento_description")


def _resolve_mesh_path(uri: str, urdf_path: Path, package_root: Path, package_name: str) -> Path:
  if uri.startswith("package://"):
    package_path = uri[len("package://") :]
    package, _, relative = package_path.partition("/")
    root = package_root if package == package_name else package_root.parent / package
    return (root / relative).resolve()
  if uri.startswith("file://"):
    return Path(uri[len("file://") :]).expanduser().resolve()
  return (urdf_path.parent / uri).resolve()


def _material_colors(robot: ET.Element) -> dict[str, tuple[float, float, float, float]]:
  colors = {}
  for material in _children(robot, "material"):
    color = _child(material, "color")
    if material.get("name") and color is not None:
      rgba = _parse_numbers(color.get("rgba"), DEFAULT_ROBOT_COLOR)
      colors[material.get("name", "")] = rgba
  return colors


def _visual_material(visual: ET.Element, colors: dict[str, tuple[float, float, float, float]]):
  material = _child(visual, "material")
  if material is None:
    return None
  color = _child(material, "color")
  if color is not None:
    return _parse_numbers(color.get("rgba"), DEFAULT_ROBOT_COLOR)
  return colors.get(material.get("name", ""))


def _read_stl(path: Path) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
  data = path.read_bytes()
  vertices: list[tuple[float, float, float]] = []
  faces: list[tuple[int, int, int]] = []
  vertex_ids: dict[tuple[float, float, float], int] = {}

  def add_vertex(point: tuple[float, float, float]) -> int:
    key = tuple(round(float(value), 8) for value in point)
    index = vertex_ids.get(key)
    if index is None:
      index = len(vertices)
      vertex_ids[key] = index
      vertices.append(tuple(float(value) for value in point))
    return index

  if len(data) >= 84:
    triangle_count = struct.unpack_from("<I", data, 80)[0]
  else:
    triangle_count = -1
  if triangle_count >= 0 and 84 + triangle_count * 50 == len(data):
    offset = 84
    for _ in range(triangle_count):
      values = struct.unpack_from("<12f", data, offset)
      offset += 50
      faces.append(
        tuple(add_vertex(tuple(values[start : start + 3])) for start in (3, 6, 9))
      )
  else:
    for line in data.decode("ascii", errors="ignore").splitlines():
      parts = line.split()
      if len(parts) == 4 and parts[0].lower() == "vertex":
        point = tuple(float(value) for value in parts[1:])
        index = add_vertex(point)
        if len(faces) == 0 or len(faces[-1]) == 3:
          faces.append((index,))
        else:
          faces[-1] = (*faces[-1], index)
    faces = [face for face in faces if len(face) == 3]
  if not faces:
    raise ValueError(f"STL contains no triangles: {path}")
  return vertices, faces


def _dae_effect_colors(root: ET.Element) -> dict[str, tuple[float, float, float, float]]:
  effects: dict[str, tuple[float, float, float, float]] = {}
  for effect in root.iter():
    if _tag(effect) != "effect" or not effect.get("id"):
      continue
    diffuse = next((item for item in effect.iter() if _tag(item) == "diffuse"), None)
    color = _child(diffuse, "color") if diffuse is not None else None
    if color is not None:
      effects[effect.get("id", "")] = _parse_numbers(color.text, DEFAULT_ROBOT_COLOR)

  materials = {}
  for material in root.iter():
    if _tag(material) != "material" or not material.get("id"):
      continue
    instance = _child(material, "instance_effect")
    if instance is not None:
      effect_id = instance.get("url", "").lstrip("#")
      if effect_id in effects:
        materials[material.get("id", "")] = effects[effect_id]
  return materials


def _dae_node_matrix(node: ET.Element, Matrix: Any, Vector: Any, Quaternion: Any):
  result = Matrix.Identity(4)
  for transform in node:
    kind = _tag(transform)
    values = tuple(float(part) for part in transform.text.split()) if transform.text else ()
    if kind == "matrix" and len(values) == 16:
      operation = Matrix(tuple(tuple(values[row * 4 + column] for column in range(4)) for row in range(4)))
    elif kind == "translate" and len(values) == 3:
      operation = Matrix.Translation(Vector(values))
    elif kind == "scale" and len(values) == 3:
      operation = Matrix.Diagonal((*values, 1.0))
    elif kind == "rotate" and len(values) == 4:
      axis = Vector(values[:3])
      if axis.length == 0:
        continue
      operation = Quaternion(axis.normalized(), math.radians(values[3])).to_matrix().to_4x4()
    else:
      continue
    result = result @ operation
  return result


def _dae_geometry_parts(
  mesh: ET.Element,
) -> tuple[dict[str, np.ndarray], dict[str, str], list[tuple[str, str, list[tuple[int, int, int]]]]]:
  sources: dict[str, np.ndarray] = {}
  for source in _children(mesh, "source"):
    array = _child(source, "float_array")
    if array is None or not array.text:
      continue
    raw = np.fromstring(array.text, sep=" ", dtype=np.float64)
    technique = _child(source, "technique_common")
    accessor = _child(technique, "accessor") if technique is not None else None
    stride = int(accessor.get("stride", "1")) if accessor is not None else 3
    count = int(accessor.get("count", str(len(raw) // stride))) if accessor is not None else len(raw) // stride
    offset = int(accessor.get("offset", "0")) if accessor is not None else 0
    if stride < 3 or offset + count * stride > len(raw):
      continue
    sources[source.get("id", "")] = raw[offset : offset + count * stride].reshape(count, stride)[:, :3]

  vertices_sources: dict[str, str] = {}
  for vertices in _children(mesh, "vertices"):
    position = next((item for item in _children(vertices, "input") if item.get("semantic") == "POSITION"), None)
    if position is not None:
      vertices_sources[vertices.get("id", "")] = position.get("source", "").lstrip("#")

  primitives = []
  for primitive in mesh:
    kind = _tag(primitive)
    if kind not in {"triangles", "polylist"}:
      continue
    inputs = _children(primitive, "input")
    vertex_input = next(
      (item for item in inputs if item.get("semantic") in {"VERTEX", "POSITION"}),
      None,
    )
    if vertex_input is None:
      continue
    source_id = vertex_input.get("source", "").lstrip("#")
    source_id = vertices_sources.get(source_id, source_id)
    offset = int(vertex_input.get("offset", "0"))
    stride = max(int(item.get("offset", "0")) for item in inputs) + 1
    p = _child(primitive, "p")
    if p is None or not p.text:
      continue
    indices = [int(value) for value in p.text.split()]
    faces = []
    if kind == "triangles":
      for start in range(0, len(indices), stride * 3):
        if start + stride * 3 <= len(indices):
          faces.append(tuple(indices[start + corner * stride + offset] for corner in range(3)))
    else:
      counts_node = _child(primitive, "vcount")
      counts = [int(value) for value in counts_node.text.split()] if counts_node is not None and counts_node.text else []
      cursor = 0
      for count in counts:
        polygon = [indices[cursor + corner * stride + offset] for corner in range(count)]
        cursor += count * stride
        for corner in range(1, count - 1):
          faces.append((polygon[0], polygon[corner], polygon[corner + 1]))
    if faces:
      primitives.append((source_id, primitive.get("material", ""), faces))
  return sources, vertices_sources, primitives


def _read_dae(path: Path, mesh_scale: tuple[float, float, float], Matrix: Any, Vector: Any, Quaternion: Any):
  root = ET.parse(path).getroot()
  asset = _child(root, "asset")
  up_axis = _child(asset, "up_axis") if asset is not None else None
  if up_axis is not None and up_axis.text and up_axis.text.strip() != "Z_UP":
    raise ValueError(f"COLLADA mesh must use Z_UP coordinates: {path}")

  geometry_by_id = {}
  for geometry in root.iter():
    if _tag(geometry) == "geometry" and geometry.get("id"):
      mesh = _child(geometry, "mesh")
      if mesh is not None:
        geometry_by_id[geometry.get("id", "")] = (
          geometry.get("name") or geometry.get("id"),
          *_dae_geometry_parts(mesh),
        )
  if not geometry_by_id:
    raise ValueError(f"COLLADA file contains no mesh geometry: {path}")

  colors_by_id = _dae_effect_colors(root)
  visual_scene = next((item for item in root.iter() if _tag(item) == "visual_scene"), None)
  instances: list[tuple[str, Any, ET.Element]] = []

  def visit(node: ET.Element, parent_matrix: Any) -> None:
    local_matrix = _dae_node_matrix(node, Matrix, Vector, Quaternion)
    world_matrix = parent_matrix @ local_matrix
    for instance in _children(node, "instance_geometry"):
      bind_map = {}
      bind_material = _child(instance, "bind_material")
      technique = _child(bind_material, "technique_common") if bind_material is not None else None
      if technique is not None:
        for binding in _children(technique, "instance_material"):
          material_id = binding.get("target", "").lstrip("#")
          bind_map[binding.get("symbol", "")] = colors_by_id.get(material_id, DEFAULT_ROBOT_COLOR)
      instances.append((instance.get("url", "").lstrip("#"), world_matrix.copy(), bind_map))
    for child_node in _children(node, "node"):
      visit(child_node, world_matrix)

  if visual_scene is not None:
    for node in _children(visual_scene, "node"):
      visit(node, Matrix.Identity(4))
  if not instances:
    instances = [(geometry_id, Matrix.Identity(4), {}) for geometry_id in geometry_by_id]

  result = []
  scale_vector = np.asarray(mesh_scale, dtype=np.float64)
  for geometry_id, transform, bound_materials in instances:
    geometry = geometry_by_id.get(geometry_id)
    if geometry is None:
      continue
    name, sources, _, primitives = geometry
    matrix = np.asarray(tuple(tuple(float(value) for value in row) for row in transform), dtype=np.float64)
    vertex_arrays: dict[str, tuple[int, int]] = {}
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    face_colors: list[tuple[float, float, float, float]] = []
    for source_id, material_symbol, primitive_faces in primitives:
      source_vertices = sources.get(source_id)
      if source_vertices is None:
        continue
      if source_id not in vertex_arrays:
        transformed = source_vertices @ matrix[:3, :3].T + matrix[:3, 3]
        transformed *= scale_vector
        offset = len(vertices)
        vertices.extend(tuple(float(component) for component in row) for row in transformed)
        vertex_arrays[source_id] = (offset, len(source_vertices))
      offset, source_count = vertex_arrays[source_id]
      color = bound_materials.get(material_symbol, DEFAULT_ROBOT_COLOR)
      for face in primitive_faces:
        if min(face) < 0 or max(face) >= source_count:
          continue
        faces.append(tuple(offset + index for index in face))
        face_colors.append(color)
    if faces:
      result.append((name, vertices, faces, face_colors))
  if not result:
    raise ValueError(f"COLLADA file contains no supported triangles: {path}")
  return result


def _origin(element: ET.Element | None):
  xyz = _parse_numbers(element.get("xyz") if element is not None else None, (0.0, 0.0, 0.0))
  rpy = _parse_numbers(element.get("rpy") if element is not None else None, (0.0, 0.0, 0.0))
  return xyz, rpy


def _description_tree(robot: ET.Element, root_link: str):
  links = {link.get("name", ""): link for link in _children(robot, "link") if link.get("name")}
  if root_link not in links:
    raise ValueError(f"root link {root_link!r} is missing from the URDF")

  children = {}
  for joint in _children(robot, "joint"):
    parent = _child(joint, "parent")
    child = _child(joint, "child")
    if parent is None or child is None:
      continue
    axis = _child(joint, "axis")
    origin = _child(joint, "origin")
    xyz, rpy = _origin(origin)
    axis_xyz = _parse_numbers(axis.get("xyz") if axis is not None else None, (1.0, 0.0, 0.0))
    description = {
      "name": joint.get("name", ""),
      "type": joint.get("type", "fixed"),
      "parent": parent.get("link", ""),
      "child": child.get("link", ""),
      "xyz": xyz,
      "rpy": rpy,
      "axis": axis_xyz,
    }
    children.setdefault(description["parent"], []).append(description)
  return links, children


def _load_capture(path: Path):
  with np.load(path, allow_pickle=False) as archive:
    capture = {key: archive[key].copy() for key in archive.files}
  required = {"time", "root_pos", "root_quat", "joint_pos"}
  missing = sorted(required - capture.keys())
  if missing:
    raise ValueError(f"capture is missing required channels: {', '.join(missing)}")
  times = np.asarray(capture["time"], dtype=np.float64)
  root_pos = np.asarray(capture["root_pos"], dtype=np.float64)
  root_quat = np.asarray(capture["root_quat"], dtype=np.float64)
  joint_pos = np.asarray(capture["joint_pos"], dtype=np.float64)
  count = len(times)
  if times.ndim != 1 or count < 2 or np.any(np.diff(times) <= 0.0):
    raise ValueError("capture time must contain at least two strictly increasing samples")
  if root_pos.shape != (count, 3) or root_quat.shape != (count, 4):
    raise ValueError("root_pos and root_quat must have shapes (frames, 3) and (frames, 4)")
  if joint_pos.ndim != 2 or joint_pos.shape[0] != count:
    raise ValueError("joint_pos must have one row per capture frame")
  if not all(np.isfinite(value).all() for value in (times, root_pos, root_quat, joint_pos)):
    raise ValueError("capture contains non-finite motion values")
  norms = np.linalg.norm(root_quat, axis=1)
  if np.any(norms < 1.0e-12):
    raise ValueError("capture contains a zero-length root quaternion")

  joint_names = DEFAULT_CAPTURE_JOINT_NAMES
  if "meta_joint_names" in capture:
    value = np.asarray(capture["meta_joint_names"])
    if value.ndim == 0:
      try:
        joint_names = tuple(json.loads(str(value.item())))
      except (TypeError, ValueError) as error:
        raise ValueError("meta_joint_names must be a JSON array of names") from error
  if len(joint_names) != joint_pos.shape[1] or len(set(joint_names)) != len(joint_names):
    raise ValueError("capture joint names do not match the joint_pos channel width")
  joint_indices = {name: index for index, name in enumerate(joint_names)}
  missing_joints = sorted(set(SIM_TO_URDF_JOINT) - joint_indices.keys())
  if missing_joints:
    raise ValueError(f"capture is missing named joint channels: {', '.join(missing_joints)}")
  return capture, times, root_pos, root_quat, joint_pos, joint_indices


def _metadata_text(capture: dict[str, np.ndarray], key: str) -> str:
  value = capture.get(key)
  if value is None or np.asarray(value).ndim != 0:
    return ""
  return str(np.asarray(value).item())


def _choose_fps(capture: dict[str, np.ndarray], times: np.ndarray, requested: float | None) -> float:
  if requested is not None:
    if requested <= 0.0:
      raise ValueError("--fps must be greater than zero")
    return float(requested)
  clip_metadata = _metadata_text(capture, "clip_metadata_json")
  if clip_metadata:
    try:
      clip_fps = json.loads(clip_metadata).get("clip_fps")
      if clip_fps and float(clip_fps) > 0.0:
        return float(clip_fps)
    except (TypeError, ValueError):
      pass
  recorded_fps = _metadata_text(capture, "meta_fps")
  if recorded_fps:
    try:
      value = float(recorded_fps)
      if value > 0.0:
        return value
    except ValueError:
      pass
  return float(1.0 / np.median(np.diff(times)))


def _event_markers(capture: dict[str, np.ndarray], times: np.ndarray) -> list[tuple[str, float]]:
  markers = []
  encoded = _metadata_text(capture, "clip_events_json")
  if encoded:
    try:
      for marker in json.loads(encoded):
        event_time = marker.get("clip_time", marker.get("source_time"))
        if event_time is not None:
          markers.append((str(marker["name"]), float(event_time)))
    except (TypeError, ValueError, KeyError):
      markers = []
  if markers:
    return markers

  jump_state = capture.get("jump_state")
  if jump_state is not None and jump_state.ndim == 2 and jump_state.shape[1] >= 3:
    for index in np.flatnonzero(jump_state[:, 1] > 0.5):
      markers.append(("takeoff", float(times[index] - times[0])))
    for index in np.flatnonzero(jump_state[:, 2] > 0.5):
      markers.append(("landing", float(times[index] - times[0])))
    if markers:
      return markers
  contacts = capture.get("contacts")
  if contacts is not None and contacts.ndim == 2 and contacts.shape[1] >= 2:
    supported = np.all(contacts[:, :2] > 0.5, axis=1)
    markers.extend(
      ("takeoff", float(times[index] - times[0]))
      for index in np.flatnonzero(supported[:-1] & ~supported[1:]) + 1
    )
    markers.extend(
      ("landing", float(times[index] - times[0]))
      for index in np.flatnonzero(~supported[:-1] & supported[1:]) + 1
    )
  return markers


def _clear_scene(bpy: Any) -> None:
  bpy.ops.wm.read_factory_settings(use_empty=True)


def _make_empty(bpy: Any, collection: Any, name: str, parent: Any = None):
  obj = bpy.data.objects.new(name, None)
  obj.empty_display_type = "PLAIN_AXES"
  obj.empty_display_size = 0.06
  collection.objects.link(obj)
  if parent is not None:
    obj.parent = parent
  return obj


def _blender_material(
  bpy: Any,
  cache: dict,
  rgba: tuple[float, float, float, float],
  role: str = "body",
):
  color = tuple(max(0.0, min(1.0, float(value))) for value in rgba)
  key = (role, *(round(value, 5) for value in color))
  if key in cache:
    return cache[key]
  material = bpy.data.materials.new(f"Ascento_{role}_{len(cache):02d}")
  material.diffuse_color = color
  material.use_nodes = True
  principled = material.node_tree.nodes.get("Principled BSDF")
  if principled is not None:
    principled.inputs["Base Color"].default_value = color
    luminance = sum(color[index] * weight for index, weight in enumerate((0.2126, 0.7152, 0.0722)))
    coat_weight = 0.0
    if role == "wheel" and luminance < 0.18:
      metallic, roughness = 0.0, 0.9
      bump_strength, bump_distance = 0.14, 0.0008
    elif role == "head" and luminance < 0.025:
      metallic, roughness = 0.0, 0.22
      coat_weight, bump_strength, bump_distance = 0.28, 0.0, 0.0
    elif role == "sensor":
      metallic, roughness = 0.12, 0.3
      coat_weight, bump_strength, bump_distance = 0.06, 0.025, 0.00015
    elif role in {"body", "leg"} and luminance >= 0.65:
      # The URDF's pale robot surfaces are painted/polymer covers, not bare
      # metal. Keep the broad highlights subdued so they sit naturally in an
      # outdoor plate.
      metallic, roughness = 0.025, 0.43
      coat_weight, bump_strength, bump_distance = 0.08, 0.035, 0.0002
    elif luminance >= 0.72:
      metallic, roughness = 0.05, 0.42
      coat_weight, bump_strength, bump_distance = 0.08, 0.035, 0.0002
    elif luminance >= 0.38:
      metallic, roughness = 0.12, 0.42
      bump_strength, bump_distance = 0.04, 0.00025
    else:
      metallic, roughness = 0.04, 0.5
      coat_weight, bump_strength, bump_distance = 0.08, 0.05, 0.0003
    principled.inputs["Metallic"].default_value = metallic
    principled.inputs["Roughness"].default_value = roughness
    if "Coat Weight" in principled.inputs:
      principled.inputs["Coat Weight"].default_value = coat_weight
      principled.inputs["Coat Roughness"].default_value = 0.26
    if "Specular IOR Level" in principled.inputs:
      principled.inputs["Specular IOR Level"].default_value = 0.35

    if bump_strength > 0.0:
      coordinates = material.node_tree.nodes.new("ShaderNodeTexCoord")
      coordinates.location = (-660.0, -160.0)
      noise = material.node_tree.nodes.new("ShaderNodeTexNoise")
      noise.location = (-430.0, -160.0)
      noise.inputs["Scale"].default_value = 260.0 if role == "wheel" else 150.0
      noise.inputs["Detail"].default_value = 2.0
      bump = material.node_tree.nodes.new("ShaderNodeBump")
      bump.location = (-180.0, -160.0)
      bump.inputs["Strength"].default_value = bump_strength
      bump.inputs["Distance"].default_value = bump_distance
      material.node_tree.links.new(coordinates.outputs["Object"], noise.inputs["Vector"])
      material.node_tree.links.new(noise.outputs["Fac"], bump.inputs["Height"])
      material.node_tree.links.new(bump.outputs["Normal"], principled.inputs["Normal"])
  cache[key] = material
  return material


def _mesh_object(
  bpy: Any,
  collection: Any,
  parent: Any,
  name: str,
  vertices: list[tuple[float, float, float]],
  faces: list[tuple[int, int, int]],
  material_cache: dict,
  colors: list[tuple[float, float, float, float]] | None = None,
  color: tuple[float, float, float, float] | None = None,
  material_role: str = "body",
):
  mesh = bpy.data.meshes.new(name)
  mesh.from_pydata(vertices, [], faces)
  mesh.update()
  obj = bpy.data.objects.new(name, mesh)
  collection.objects.link(obj)
  obj.parent = parent
  obj["ascento_visual_mesh"] = True
  palette = []
  palette_indices = {}
  face_material_indices = []
  source_colors = colors if colors is not None else [color or DEFAULT_ROBOT_COLOR] * len(faces)
  for rgba in source_colors:
    rgba_key = tuple(round(float(value), 5) for value in rgba)
    if rgba_key not in palette_indices:
      palette_indices[rgba_key] = len(palette)
      palette.append(_blender_material(bpy, material_cache, rgba, role=material_role))
    face_material_indices.append(palette_indices[rgba_key])
  for material in palette:
    mesh.materials.append(material)
  for polygon, material_index in zip(mesh.polygons, face_material_indices):
    polygon.material_index = material_index
    polygon.use_smooth = True
  return obj


def _build_visuals(
  bpy: Any,
  link_name: str,
  link_element: ET.Element,
  link_object: Any,
  urdf_path: Path,
  package_root: Path,
  package_name: str,
  colors: dict[str, tuple[float, float, float, float]],
  collection: Any,
  material_cache: dict,
  Matrix: Any,
  Vector: Any,
  Quaternion: Any,
) -> None:
  from mathutils import Euler

  normalized_link_name = link_name.lower()
  if "wheel" in normalized_link_name:
    material_role = "wheel"
  elif "head" in normalized_link_name:
    material_role = "head"
  elif "lidar" in normalized_link_name or "camera" in normalized_link_name:
    material_role = "sensor"
  else:
    material_role = "leg"

  for visual_index, visual in enumerate(_children(link_element, "visual")):
    origin = _child(visual, "origin")
    xyz, rpy = _origin(origin)
    rotation = Euler(rpy, "XYZ").to_quaternion()
    geometry = _child(visual, "geometry")
    if geometry is None:
      continue
    mesh_element = _child(geometry, "mesh")
    if mesh_element is None:
      continue
    filename = mesh_element.get("filename")
    if not filename:
      continue
    mesh_path = _resolve_mesh_path(filename, urdf_path, package_root, package_name)
    if not mesh_path.is_file():
      raise FileNotFoundError(f"visual mesh referenced by URDF was not found: {mesh_path}")
    mesh_scale = _parse_numbers(mesh_element.get("scale"), (1.0, 1.0, 1.0))
    override_color = _visual_material(visual, colors)
    stem = f"Ascento_{link_name.replace('/', '_')}_visual_{visual_index:02d}"
    if mesh_path.suffix.lower() == ".stl":
      raw_vertices, faces = _read_stl(mesh_path)
      vertices = [
        (point[0] * mesh_scale[0], point[1] * mesh_scale[1], point[2] * mesh_scale[2])
        for point in raw_vertices
      ]
      obj = _mesh_object(
        bpy,
        collection,
        link_object,
        stem,
        vertices,
        faces,
        material_cache,
        color=override_color,
        material_role=material_role,
      )
      obj["ascento_link_name"] = link_name
      obj.location = xyz
      obj.rotation_mode = "QUATERNION"
      obj.rotation_quaternion = rotation
    elif mesh_path.suffix.lower() == ".dae":
      dae_parts = _read_dae(mesh_path, mesh_scale, Matrix, Vector, Quaternion)
      for part_index, (_, vertices, faces, face_colors) in enumerate(dae_parts):
        obj = _mesh_object(
          bpy,
          collection,
          link_object,
          f"{stem}_{part_index:02d}",
          vertices,
          faces,
          material_cache,
          colors=[override_color] * len(faces) if override_color is not None else face_colors,
          material_role=material_role,
        )
        obj["ascento_link_name"] = link_name
        obj.location = xyz
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = rotation
    else:
      raise ValueError(f"unsupported URDF visual mesh format: {mesh_path.suffix}")


def _build_robot(
  bpy: Any,
  urdf_path: Path,
  package_root: Path,
  root_link: str,
  collection: Any,
  material_cache: dict,
  Matrix: Any,
  Vector: Any,
  Quaternion: Any,
):
  from mathutils import Euler

  robot = ET.parse(urdf_path).getroot()
  links, children = _description_tree(robot, root_link)
  package_name = _package_name(package_root, robot)
  colors = _material_colors(robot)
  animated_joints = {}
  root_object = _make_empty(bpy, collection, "Ascento_Root")
  root_object["urdf_root_link"] = root_link

  def build_link(link_name: str, link_object: Any) -> None:
    _build_visuals(
      bpy,
      link_name,
      links[link_name],
      link_object,
      urdf_path,
      package_root,
      package_name,
      colors,
      collection,
      material_cache,
      Matrix,
      Vector,
      Quaternion,
    )
    for joint in children.get(link_name, []):
      child_name = joint["child"]
      if child_name not in links:
        continue
      pivot = _make_empty(
        bpy,
        collection,
        f"Ascento_Joint_{joint['name'].replace('/', '_')}",
        link_object,
      )
      pivot.location = joint["xyz"]
      pivot.rotation_mode = "QUATERNION"
      pivot.rotation_quaternion = Euler(joint["rpy"], "XYZ").to_quaternion()
      child_object = _make_empty(
        bpy,
        collection,
        f"Ascento_Link_{child_name.replace('/', '_')}",
        pivot,
      )
      sim_joint = next((name for name, urdf_name in SIM_TO_URDF_JOINT.items() if urdf_name == joint["name"]), None)
      if joint["type"] in {"revolute", "continuous"} and sim_joint is not None:
        axis = Vector(joint["axis"])
        if axis.length == 0:
          raise ValueError(f"URDF joint has a zero-length axis: {joint['name']}")
        axis.normalize()
        child_object.rotation_mode = "AXIS_ANGLE"
        child_object.rotation_axis_angle = (0.0, axis.x, axis.y, axis.z)
        animated_joints[sim_joint] = (child_object, tuple(axis))
      elif joint["type"] == "prismatic":
        raise ValueError(f"prismatic joints are not supported by the Ascento capture mapping: {joint['name']}")
      elif joint["type"] not in {"fixed", "revolute", "continuous"}:
        raise ValueError(f"unsupported URDF joint type {joint['type']!r}: {joint['name']}")
      build_link(child_name, child_object)

  build_link(root_link, root_object)
  missing = sorted(set(SIM_TO_URDF_JOINT) - animated_joints.keys())
  if missing:
    raise ValueError(f"URDF root subtree is missing animated joints: {', '.join(missing)}")
  return root_object, animated_joints, robot.get("name", "ascento")


def _keyframe_motion(
  root_object: Any,
  animated_joints: dict[str, tuple[Any, tuple[float, float, float]]],
  times: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  joint_pos: np.ndarray,
  joint_indices: dict[str, int],
  fps: float,
):
  start_frame = 1
  animated_objects = [root_object]
  animated_objects.extend(item[0] for item in animated_joints.values())
  previous_quat = None
  for frame_index, source_time in enumerate(times):
    frame = start_frame + (float(source_time) - float(times[0])) * fps
    root_object.location = tuple(float(value) for value in root_pos[frame_index])
    quat = np.asarray(root_quat[frame_index], dtype=np.float64)
    quat = quat / np.linalg.norm(quat)
    if previous_quat is not None and float(np.dot(previous_quat, quat)) < 0.0:
      quat = -quat
    previous_quat = quat
    root_object.rotation_mode = "QUATERNION"
    root_object.rotation_quaternion = tuple(float(value) for value in quat)
    root_object.keyframe_insert(data_path="location", frame=frame)
    root_object.keyframe_insert(data_path="rotation_quaternion", frame=frame)
    for sim_name, (obj, axis) in animated_joints.items():
      obj.rotation_axis_angle = (float(joint_pos[frame_index, joint_indices[sim_name]]), *axis)
      obj.keyframe_insert(data_path="rotation_axis_angle", frame=frame)
  return start_frame, int(math.ceil(start_frame + (float(times[-1]) - float(times[0])) * fps))


def _make_camera_and_lights(bpy: Any, scene: Any, center: tuple[float, float, float], scale: float):
  from mathutils import Vector

  target = Vector(center)
  camera_data = bpy.data.cameras.new("Ascento_Camera")
  camera_data.type = "ORTHO"
  camera_data.ortho_scale = scale
  camera = bpy.data.objects.new("Ascento_Camera", camera_data)
  scene.collection.objects.link(camera)
  camera.location = target + Vector((3.2, -5.5, 0.6))
  camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
  scene.camera = camera


def _configure_cycles(scene: Any) -> None:
  scene.render.engine = "CYCLES"
  scene.cycles.samples = 32
  scene.cycles.preview_samples = 16
  scene.cycles.use_denoising = True
  scene.cycles.use_preview_denoising = True


def _camera_shot_ranges(
  start_frame: int,
  end_frame: int,
  shot_names: list[str],
) -> list[tuple[str, int, int]]:
  """Divide an inclusive frame range into contiguous, balanced camera cuts."""
  if not shot_names:
    return []
  unknown = sorted(set(shot_names) - set(CINEMATIC_SHOTS))
  if unknown:
    raise ValueError(f"unknown camera shot(s): {', '.join(unknown)}; choose from {', '.join(CINEMATIC_SHOTS)}")
  frame_count = end_frame - start_frame + 1
  if frame_count < len(shot_names):
    raise ValueError(
      f"camera plan needs at least one frame per shot ({len(shot_names)} shots, {frame_count} frames)"
    )
  frames_per_shot, extra_frames = divmod(frame_count, len(shot_names))
  result = []
  frame = start_frame
  for index, name in enumerate(shot_names):
    duration = frames_per_shot + (1 if index < extra_frames else 0)
    shot_end = frame + duration - 1
    result.append((name, frame, shot_end))
    frame = shot_end + 1
  return result


def _sample_root_pose(
  frame: int,
  start_frame: int,
  times: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  fps: float,
  Quaternion: Any,
):
  source_time = float(times[0]) + (frame - start_frame) / fps
  position = np.asarray(
    [np.interp(source_time, times, root_pos[:, axis]) for axis in range(3)], dtype=np.float64
  )
  right = int(np.searchsorted(times, source_time, side="right"))
  if right <= 0:
    quaternion = Quaternion(tuple(float(value) for value in root_quat[0]))
  elif right >= len(times):
    quaternion = Quaternion(tuple(float(value) for value in root_quat[-1]))
  else:
    left = right - 1
    interval = float(times[right] - times[left])
    alpha = (source_time - float(times[left])) / interval
    first = Quaternion(tuple(float(value) for value in root_quat[left])).normalized()
    second = Quaternion(tuple(float(value) for value in root_quat[right])).normalized()
    quaternion = first.slerp(second, alpha)
  quaternion.normalize()
  return position, quaternion


def _robot_bounds_center(bpy: Any, scene: Any, fallback_position: np.ndarray):
  from mathutils import Vector

  visual_meshes = [obj for obj in scene.objects if obj.type == "MESH" and obj.get("ascento_visual_mesh")]
  if not visual_meshes:
    return Vector(tuple(float(value) for value in fallback_position)) + Vector((0.0, 0.0, 0.36))

  bpy.context.view_layer.update()
  depsgraph = bpy.context.evaluated_depsgraph_get()
  corners = [
    obj.matrix_world @ Vector(corner)
    for source in visual_meshes
    for obj in (source.evaluated_get(depsgraph),)
    for corner in obj.bound_box
  ]
  low = tuple(min(point[axis] for point in corners) for axis in range(3))
  high = tuple(max(point[axis] for point in corners) for axis in range(3))
  return Vector(tuple((low[axis] + high[axis]) * 0.5 for axis in range(3)))


def _wheel_contact_pair(bpy: Any, scene: Any):
  """Return the left/right wheel bottom centers in evaluated world space."""
  from mathutils import Vector

  by_side: dict[str, list[Any]] = {"left": [], "right": []}
  for obj in scene.objects:
    name = str(obj.get("ascento_link_name", "")).lower()
    if obj.type != "MESH" or not obj.get("ascento_visual_mesh") or "wheel" not in name:
      continue
    for side in by_side:
      if side in name:
        by_side[side].append(obj)
  if not all(by_side.values()):
    return None

  bpy.context.view_layer.update()
  depsgraph = bpy.context.evaluated_depsgraph_get()
  bottoms = {}
  for side, sources in by_side.items():
    points = [
      obj.matrix_world @ Vector(corner)
      for source in sources
      for obj in (source.evaluated_get(depsgraph),)
      for corner in obj.bound_box
    ]
    lowest = min(point.z for point in points)
    edge = [point for point in points if point.z <= lowest + 1.0e-5]
    bottoms[side] = Vector(tuple(sum(point[axis] for point in edge) / len(edge) for axis in range(3)))
  return bottoms["left"], bottoms["right"]


def _contact_bisector_heading(forward: Any, left_point: Any, right_point: Any, Vector: Any):
  """Choose the front-facing horizontal view direction equidistant to both wheels."""
  baseline = Vector((left_point.x - right_point.x, left_point.y - right_point.y, 0.0))
  heading = Vector((forward.x, forward.y, 0.0))
  if baseline.length < 0.05 or heading.length < 1.0e-8:
    return heading.normalized() if heading.length else Vector((1.0, 0.0, 0.0))
  candidate = Vector((-baseline.y, baseline.x, 0.0)).normalized()
  heading.normalize()
  if candidate.dot(heading) < 0.0:
    candidate.negate()
  return candidate


def _create_cinematic_cameras(
  bpy: Any,
  scene: Any,
  shot_names: list[str],
  times: np.ndarray,
  root_pos: np.ndarray,
  root_quat: np.ndarray,
  fps: float,
  start_frame: int,
  end_frame: int,
  depth_of_field: bool = False,
) -> list[tuple[str, int, int, Any]]:
  from mathutils import Quaternion, Vector

  lens_by_shot = {"low_front": 52.0, "side_follow": 45.0, "rear_chase": 38.0, "orbit": 50.0}
  cuts = []
  vertical = Vector((0.0, 0.0, 1.0))
  focus_target = None
  if depth_of_field:
    focus_target = _make_empty(bpy, scene.collection, "Ascento_Cinematic_Focus_Target")
    focus_target.empty_display_size = 0.04
  for cut_index, (shot_name, cut_start, cut_end) in enumerate(
    _camera_shot_ranges(start_frame, end_frame, shot_names), start=1
  ):
    camera_data = bpy.data.cameras.new(f"Ascento_Camera_{shot_name}")
    camera_data.lens = lens_by_shot[shot_name]
    camera_data.dof.use_dof = depth_of_field
    camera_data.dof.aperture_fstop = 8.0
    if focus_target is not None:
      camera_data.dof.focus_object = focus_target
    camera = bpy.data.objects.new(f"Ascento_Camera_{shot_name}", camera_data)
    scene.collection.objects.link(camera)
    marker = scene.timeline_markers.new(f"CUT_{cut_index:02d}_{shot_name}", frame=cut_start)
    marker.camera = camera
    cut_span = max(1, cut_end - cut_start)
    low_front_heading_offset = None

    for frame in range(cut_start, cut_end + 1):
      scene.frame_set(frame)
      progress = (frame - cut_start) / cut_span
      position, rotation = _sample_root_pose(
        frame, start_frame, times, root_pos, root_quat, fps, Quaternion
      )
      target = _robot_bounds_center(bpy, scene, position)
      forward = rotation @ Vector((1.0, 0.0, 0.0))
      forward.z = 0.0
      if forward.length < 1.0e-8:
        forward = Vector((1.0, 0.0, 0.0))
      else:
        forward.normalize()
      left = Vector((-forward.y, forward.x, 0.0))

      if shot_name == "low_front":
        # Aim along the perpendicular bisector of the wheel supports. This
        # keeps both real ground contacts at the same camera depth, avoiding
        # the perspective cue that made the far wheel look suspended.
        wheel_pair = _wheel_contact_pair(bpy, scene)
        if wheel_pair is not None:
          left_contact, right_contact = wheel_pair
          bisector_heading = _contact_bisector_heading(forward, left_contact, right_contact, Vector)
          desired_offset = math.atan2(bisector_heading.dot(left), bisector_heading.dot(forward))
          if low_front_heading_offset is None:
            low_front_heading_offset = desired_offset
          else:
            while desired_offset - low_front_heading_offset > math.pi:
              desired_offset -= 2.0 * math.pi
            while desired_offset - low_front_heading_offset < -math.pi:
              desired_offset += 2.0 * math.pi
            # Smooth pose-driven changes so the camera does not twitch with
            # each small change in leg posture.
            low_front_heading_offset += 0.08 * (desired_offset - low_front_heading_offset)
          camera_forward = forward * math.cos(low_front_heading_offset) + left * math.sin(low_front_heading_offset)
          contact_axis = Vector((left_contact.x - right_contact.x, left_contact.y - right_contact.y, 0.0)).normalized()
          contact_midpoint = (left_contact + right_contact) * 0.5
          toward_midpoint = Vector((contact_midpoint.x - target.x, contact_midpoint.y - target.y, 0.0))
          bisector_shift = contact_axis * toward_midpoint.dot(contact_axis)
          offset = camera_forward * 3.8 + bisector_shift + vertical * 0.6
        else:
          offset = forward * 3.8 - left * 0.12 + vertical * 0.6
      elif shot_name == "side_follow":
        offset = -forward * 0.45 + left * 3.4 + vertical * 0.9
      elif shot_name == "rear_chase":
        offset = -forward * 4.1 - left * 0.55 + vertical * 1.05
      else:
        angle = math.radians(-68.0 + 142.0 * progress)
        offset = (forward * math.cos(angle) + left * math.sin(angle)) * 4.3 + vertical * 0.95

      camera.location = target + offset
      camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
      camera.keyframe_insert(data_path="location", frame=frame)
      camera.keyframe_insert(data_path="rotation_euler", frame=frame)
      if focus_target is not None:
        focus_target.location = target
        focus_target.keyframe_insert(data_path="location", frame=frame)
    _set_linear_animation([camera])
    cuts.append((shot_name, cut_start, cut_end, camera))

  if focus_target is not None:
    _set_linear_animation([focus_target])
  if cuts:
    scene.frame_set(start_frame)
    scene.camera = cuts[0][3]
  return cuts


def _resolve_warehouse_fbx(warehouse_asset: Path, extraction_root: Path) -> Path:
  source = warehouse_asset.expanduser().resolve()
  if not source.exists():
    raise FileNotFoundError(
      f"warehouse asset does not exist: {source}; place the Sketchfab FBX/ZIP package there or pass --warehouse"
    )
  if source.suffix.lower() == ".zip":
    _safe_extract_zip(source, extraction_root)
    search_root = extraction_root
  elif source.is_dir():
    search_root = source
  elif source.suffix.lower() == ".fbx":
    return source
  else:
    raise ValueError("--warehouse must be an FBX file, asset directory, or ZIP package")

  candidates = sorted(search_root.rglob("*.fbx"))
  if not candidates:
    raise FileNotFoundError(f"no FBX model found in warehouse package: {search_root}")
  named = [candidate for candidate in candidates if "warehouse" in candidate.stem.lower()]
  pool = named or candidates
  largest_size = max(candidate.stat().st_size for candidate in pool)
  largest = [candidate for candidate in pool if candidate.stat().st_size == largest_size]
  if len(largest) != 1:
    listing = ", ".join(str(candidate.relative_to(search_root)) for candidate in largest)
    raise ValueError(f"warehouse package contains multiple equally sized FBX models; pass one directly: {listing}")
  return largest[0]


def _make_warehouse_stage(
  bpy: Any,
  scene: Any,
  warehouse_asset: Path,
  extraction_root: Path,
  center: tuple[float, float, float],
  motion_span: float,
) -> Any:
  """Import and ground the Sketchfab warehouse as the actual 3D environment."""
  from mathutils import Vector

  model_path = _resolve_warehouse_fbx(warehouse_asset, extraction_root)
  existing = set(bpy.data.objects)
  bpy.ops.import_scene.fbx(filepath=str(model_path), use_image_search=True)
  imported = [obj for obj in bpy.data.objects if obj not in existing and obj.type not in {"CAMERA", "LIGHT"}]
  meshes = [obj for obj in imported if obj.type == "MESH"]
  if not meshes:
    raise ValueError(f"warehouse FBX contains no mesh geometry: {model_path}")

  bpy.context.view_layer.update()
  depsgraph = bpy.context.evaluated_depsgraph_get()
  bounds = [
    evaluated.matrix_world @ Vector(corner)
    for source in meshes
    for evaluated in (source.evaluated_get(depsgraph),)
    for corner in evaluated.bound_box
  ]
  low = Vector(tuple(min(point[axis] for point in bounds) for axis in range(3)))
  high = Vector(tuple(max(point[axis] for point in bounds) for axis in range(3)))
  dimensions = high - low
  horizontal_span = max(dimensions.x, dimensions.y)
  if horizontal_span <= 1.0e-5:
    raise ValueError(f"warehouse FBX has no usable horizontal extent: {model_path}")

  stage = bpy.data.objects.new("Ascento_Warehouse_Stage", None)
  stage.empty_display_type = "CUBE"
  scene.collection.objects.link(stage)
  world_matrices = {obj: obj.matrix_world.copy() for obj in imported}
  for obj in imported:
    obj.parent = stage
    obj.matrix_world = world_matrices[obj]

  target_span = max(24.0, float(motion_span) + 16.0)
  uniform_scale = target_span / horizontal_span
  stage.scale = (uniform_scale, uniform_scale, uniform_scale)
  stage.location = (
    float(center[0]) - (low.x + high.x) * 0.5 * uniform_scale,
    float(center[1]) - (low.y + high.y) * 0.5 * uniform_scale,
    -low.z * uniform_scale,
  )
  stage["ascento_asset_name"] = "Warehouse FBX Model Free"
  stage["ascento_asset_creator"] = "Nicholas-3D (Sketchfab: Nicholas01)"
  stage["ascento_asset_license"] = "CC BY 4.0"
  scene["ascento_warehouse_source"] = "https://sketchfab.com/3d-models/warehouse-fbx-model-free-daa7fd3ff88945298d00045ca40a4c03"
  scene["ascento_warehouse_creator"] = "Nicholas-3D"
  scene["ascento_warehouse_license"] = "CC BY 4.0"
  scene["ascento_warehouse_scale"] = uniform_scale

  world = scene.world
  if world is None:
    world = bpy.data.worlds.new("Ascento_Warehouse_World")
    scene.world = world
  world.use_nodes = True
  nodes = world.node_tree.nodes
  nodes.clear()
  background = nodes.new("ShaderNodeBackground")
  background.name = "Ascento_Warehouse_Ambience"
  background.inputs["Color"].default_value = (0.16, 0.17, 0.18, 1.0)
  background.inputs["Strength"].default_value = 0.18
  output = nodes.new("ShaderNodeOutputWorld")
  world.node_tree.links.new(background.outputs["Background"], output.inputs["Surface"])

  target = Vector(center)
  light_specs = (
    ("Ceiling_Key", 1500.0, 1.3, (1.0, 0.97, 0.92), (0.0, -1.0, 4.2)),
    ("Ceiling_Fill", 360.0, 1.8, (0.94, 0.97, 1.0), (-2.8, 0.4, 3.4)),
    ("Warehouse_Rim", 420.0, 1.0, (1.0, 0.99, 0.96), (2.2, 2.3, 4.0)),
  )
  for name, energy, size, color, offset in light_specs:
    light_data = bpy.data.lights.new(f"Ascento_{name}", "AREA")
    light_data.energy = energy
    light_data.shape = "RECTANGLE"
    light_data.size = size
    light_data.size_y = size * 0.62
    light_data.color = color
    light = bpy.data.objects.new(f"Ascento_{name}", light_data)
    scene.collection.objects.link(light)
    light.location = target + Vector(offset)
    light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()

  for camera in (obj for obj in scene.objects if obj.type == "CAMERA"):
    camera.data.clip_end = max(camera.data.clip_end, target_span * 4.0)
  for image in bpy.data.images:
    if image.source == "FILE" and image.packed_file is None:
      try:
        image.pack()
      except (RuntimeError, OSError):
        pass
  scene.render.film_transparent = False
  print(f"Imported warehouse stage: {model_path} | scale {uniform_scale:.5g} | width {target_span:.2f} m")
  return stage


def _apply_wheel_floor_clearance(
  bpy: Any,
  scene: Any,
  root_object: Any,
  start_frame: int,
  end_frame: int,
  floor_z: float = 0.0,
  contacts: np.ndarray | None = None,
  times: np.ndarray | None = None,
  fps: float | None = None,
  assume_grounded: bool = False,
):
  """Keep supported wheel visuals on the floor without flattening airborne poses."""
  from mathutils import Matrix, Quaternion, Vector

  wheel_meshes = [
    obj
    for obj in scene.objects
    if obj.type == "MESH"
    and obj.get("ascento_visual_mesh")
    and "wheel" in str(obj.get("ascento_link_name", "")).lower()
  ]
  if not wheel_meshes:
    return None

  wheels_by_side = {
    side: [wheel for wheel in wheel_meshes if side in str(wheel.get("ascento_link_name", "")).lower()]
    for side in ("left", "right")
  }
  has_side_pair = all(wheels_by_side.values())

  clearance = _make_empty(bpy, scene.collection, "Ascento_Floor_Clearance")
  root_object.parent = clearance
  root_object.matrix_parent_inverse = Matrix.Identity(4)
  frame_corrections = []
  for frame in range(start_frame, end_frame + 1):
    scene.frame_set(frame)
    clearance.location = (0.0, 0.0, 0.0)
    clearance.rotation_mode = "QUATERNION"
    clearance.rotation_quaternion = Quaternion((1.0, 0.0, 0.0, 0.0))
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    root_matrix = root_object.evaluated_get(depsgraph).matrix_world.copy()
    root_inverse = root_matrix.inverted()

    def local_wheel_points(source_objects: list[Any]) -> list[Vector]:
      return [
        root_inverse @ (wheel.evaluated_get(depsgraph).matrix_world @ Vector(corner))
        for source in source_objects
        for wheel in (source.evaluated_get(depsgraph),)
        for corner in wheel.bound_box
      ]

    supported = (False, False)
    if contacts is not None and contacts.ndim == 2 and contacts.shape[1] >= 2 and times is not None and len(times):
      source_time = float(times[0]) + (frame - start_frame) / max(float(fps or 1.0), 1.0e-8)
      contact_index = int(np.clip(np.searchsorted(times, source_time), 0, len(times) - 1))
      previous_index = max(0, contact_index - 1)
      if abs(float(times[previous_index]) - source_time) < abs(float(times[contact_index]) - source_time):
        contact_index = previous_index
      supported = tuple(bool(value > 0.5) for value in contacts[contact_index, :2])
    elif assume_grounded:
      supported = (True, True)

    correction_rotation = Quaternion((1.0, 0.0, 0.0, 0.0))
    support_points: list[Vector] = []
    if has_side_pair and all(supported):
      left_points = local_wheel_points(wheels_by_side["left"])
      right_points = local_wheel_points(wheels_by_side["right"])

      def wheel_bottom(points: list[Vector], roll: float) -> float:
        roll_rotation = Quaternion((1.0, 0.0, 0.0), roll)
        return min((root_matrix @ (roll_rotation @ point)).z for point in points)

      # Solve the roll against the actual world-space wheel bottoms. A
      # root-local height comparison gets the sign wrong whenever the captured
      # root already leans, which is exactly when one wheel appears suspended.
      limit = math.radians(12.0)
      lower, upper = -limit, limit
      lower_delta = wheel_bottom(left_points, lower) - wheel_bottom(right_points, lower)
      upper_delta = wheel_bottom(left_points, upper) - wheel_bottom(right_points, upper)
      if lower_delta * upper_delta <= 0.0:
        for _iteration in range(32):
          middle = (lower + upper) * 0.5
          middle_delta = wheel_bottom(left_points, middle) - wheel_bottom(right_points, middle)
          if lower_delta * middle_delta <= 0.0:
            upper = middle
            upper_delta = middle_delta
          else:
            lower = middle
            lower_delta = middle_delta
        roll = (lower + upper) * 0.5
      else:
        roll = min(
          (-limit, 0.0, limit),
          key=lambda angle: abs(wheel_bottom(left_points, angle) - wheel_bottom(right_points, angle)),
        )
      correction_rotation = Quaternion((1.0, 0.0, 0.0), roll)
      support_points = left_points + right_points
    elif has_side_pair and any(supported):
      side = "left" if supported[0] else "right"
      support_points = local_wheel_points(wheels_by_side[side])
    elif assume_grounded and not has_side_pair:
      support_points = local_wheel_points(wheel_meshes)

    if support_points:
      # Evaluate the corrected wheel points in world space. Adding only their
      # root-local z to the root's z ignored existing root pitch/roll and could
      # leave one side several centimetres above the ground.
      corrected_world_points = [root_matrix @ (correction_rotation @ point) for point in support_points]
      vertical_correction = floor_z - min(point.z for point in corrected_world_points)
    else:
      all_points = local_wheel_points(wheel_meshes)
      lowest_world_z = min((root_matrix @ point).z for point in all_points)
      vertical_correction = max(0.0, floor_z - lowest_world_z)

    root_position = root_matrix.translation.copy()
    root_rotation = root_matrix.to_quaternion().normalized()
    world_rotation = root_rotation @ correction_rotation @ root_rotation.conjugated()
    parent_position = root_position - (world_rotation @ root_position) + Vector((0.0, 0.0, vertical_correction))
    frame_corrections.append((frame, parent_position, world_rotation))

  for frame, position, rotation in frame_corrections:
    clearance.location = position
    clearance.rotation_mode = "QUATERNION"
    clearance.rotation_quaternion = rotation
    clearance.keyframe_insert(data_path="location", frame=frame)
    clearance.keyframe_insert(data_path="rotation_quaternion", frame=frame)
  _set_linear_animation([clearance])
  scene.frame_set(start_frame)
  return clearance


def _set_linear_animation(objects: list[Any]) -> None:
  for obj in objects:
    animation_data = obj.animation_data
    action = animation_data.action if animation_data is not None else None
    curves = list(getattr(action, "fcurves", ()))
    for layer in getattr(action, "layers", ()):
      for strip in getattr(layer, "strips", ()):
        for channelbag in getattr(strip, "channelbags", ()):
          curves.extend(getattr(channelbag, "fcurves", ()))
    for curve in curves:
      for point in curve.keyframe_points:
        point.interpolation = "LINEAR"


def _configure_render(
  bpy: Any,
  scene: Any,
  center: tuple[float, float, float],
  scale: float,
  resolution: int,
  cinematic: bool = False,
):
  _configure_cycles(scene)
  scene.render.resolution_x = resolution
  scene.render.resolution_y = round(resolution * 9 / 16) if cinematic else resolution
  scene.render.resolution_percentage = 100
  scene.render.film_transparent = False
  scene.render.image_settings.file_format = "PNG"
  scene.render.image_settings.color_mode = "RGBA"
  scene.render.image_settings.compression = 15
  scene.view_settings.view_transform = "AgX" if cinematic else "Standard"
  try:
    scene.view_settings.look = "AgX - Medium High Contrast" if cinematic else "Medium High Contrast"
  except (TypeError, ValueError):
    pass
  if scene.world is None:
    scene.world = bpy.data.worlds.new("Ascento_World")
  scene.world.use_nodes = True
  background = scene.world.node_tree.nodes.get("Background")
  if background is not None:
    background.inputs["Color"].default_value = (
      (0.012, 0.020, 0.038, 1.0) if cinematic else (0.16, 0.18, 0.22, 1.0)
    )
    background.inputs["Strength"].default_value = 0.45 if cinematic else 0.7
  if not cinematic:
    _make_camera_and_lights(bpy, scene, center, scale)


def _add_event_markers(scene: Any, capture: dict[str, np.ndarray], times: np.ndarray, fps: float, start_frame: int) -> None:
  counts: dict[str, int] = {}
  for name, event_time in _event_markers(capture, times):
    counts[name] = counts.get(name, 0) + 1
    suffix = f"_{counts[name]:02d}" if counts[name] > 1 else ""
    scene.timeline_markers.new(f"{name}{suffix}", frame=start_frame + round(event_time * fps))


def run(args: argparse.Namespace) -> None:
  try:
    import bpy
    from mathutils import Matrix, Quaternion, Vector
  except ImportError as error:
    raise RuntimeError("run this script with Blender's bundled Python interpreter") from error

  capture_path = args.capture.expanduser().resolve()
  if not capture_path.is_file():
    raise FileNotFoundError(f"capture file does not exist: {capture_path}")
  output_path = args.output.expanduser().resolve()
  output_path.parent.mkdir(parents=True, exist_ok=True)
  capture, times, root_pos, root_quat, joint_pos, joint_indices = _load_capture(capture_path)
  fps = _choose_fps(capture, times, args.fps)
  if args.camera_shots:
    clip_end_frame = max(2, int(math.ceil(1 + (float(times[-1]) - float(times[0])) * fps)))
    _camera_shot_ranges(1, clip_end_frame, args.camera_shots)

  temporary_directory = tempfile.TemporaryDirectory(prefix="ascento_blender_pipeline_")
  try:
    urdf_path, package_root = _description_paths(args.description, Path(temporary_directory.name))
    _clear_scene(bpy)
    scene = bpy.context.scene
    scene.render.fps = max(1, round(fps))
    scene.render.fps_base = scene.render.fps / fps
    collection = bpy.data.collections.new("Ascento Robot")
    scene.collection.children.link(collection)
    material_cache = {}
    root_object, animated_joints, robot_name = _build_robot(
      bpy,
      urdf_path,
      package_root,
      args.root_link,
      collection,
      material_cache,
      Matrix,
      Vector,
      Quaternion,
    )
    start_frame, end_frame = _keyframe_motion(
      root_object,
      animated_joints,
      times,
      root_pos,
      root_quat,
      joint_pos,
      joint_indices,
      fps,
    )
    scene.frame_start = start_frame
    scene.frame_end = max(start_frame + 1, end_frame)
    _add_event_markers(scene, capture, times, fps, start_frame)
    animated_objects = [root_object, *(item[0] for item in animated_joints.values())]
    _set_linear_animation(animated_objects)
    scene.frame_set(start_frame)
    scene["ascento_robot"] = robot_name
    scene["ascento_source_capture"] = capture_path.name
    scene["ascento_source_urdf"] = urdf_path.name
    scene["ascento_fps"] = fps
    scene["ascento_frame_count"] = len(times)
    for key in ("task", "seed", "checkpoint", "model_sha256", "physics_profile"):
      value = _metadata_text(capture, f"meta_{key}")
      if value:
        scene[f"ascento_{key}"] = value

    floor_clearance = _apply_wheel_floor_clearance(
      bpy,
      scene,
      root_object,
      start_frame,
      scene.frame_end,
      contacts=capture.get("contacts"),
      times=times,
      fps=fps,
      assume_grounded=any(
        label in _metadata_text(capture, "meta_task").lower()
        for label in ("velocity", "balance")
      ),
    )
    if floor_clearance is not None:
      scene["ascento_floor_clearance"] = "Supported wheels are levelled to the z=0 ground plane"
      print(f"Applied animated wheel ground contact: {floor_clearance.name}")

    span_x = float(np.ptp(root_pos[:, 0]))
    span_y = float(np.ptp(root_pos[:, 1]))
    span_z = float(np.ptp(root_pos[:, 2]))
    camera_scale = max(2.2, math.hypot(span_x, span_y) + 1.8, span_z + 2.0)
    camera_center = (
      float((root_pos[:, 0].min() + root_pos[:, 0].max()) / 2.0),
      float((root_pos[:, 1].min() + root_pos[:, 1].max()) / 2.0),
      float(root_pos[:, 2].mean() - 0.12),
    )
    cinematic = bool(args.camera_shots)
    _configure_render(bpy, scene, camera_center, camera_scale, args.resolution, cinematic=cinematic)
    _make_warehouse_stage(
      bpy,
      scene,
      args.warehouse,
      Path(temporary_directory.name) / "warehouse_asset",
      camera_center,
      math.hypot(span_x, span_y),
    )
    bpy.context.view_layer.update()
    bpy.ops.file.pack_all()
    if cinematic:
      cuts = _create_cinematic_cameras(
        bpy,
        scene,
        args.camera_shots,
        times,
        root_pos,
        root_quat,
        fps,
        start_frame,
        scene.frame_end,
        depth_of_field=args.cinematic_dof,
      )
      scene["ascento_camera_shots"] = json.dumps(args.camera_shots)
      scene["ascento_cinematic_dof"] = args.cinematic_dof
      print(
        "Cinematic camera cuts: "
        + ", ".join(f"{name} [{first}-{last}]" for name, first, last, _ in cuts)
      )
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))
    print(f"Saved animated Blender scene: {output_path}")

    if args.render_dir is not None:
      render_dir = args.render_dir.expanduser().resolve()
      render_dir.mkdir(parents=True, exist_ok=True)
      scene.render.filepath = str(render_dir / "frame_")
      bpy.ops.render.render(animation=True)
      print(f"Rendered PNG sequence: {render_dir}")
  finally:
    temporary_directory.cleanup()


def _parse_args() -> argparse.Namespace:
  try:
    separator = sys.argv.index("--")
    arguments = sys.argv[separator + 1 :]
  except ValueError:
    arguments = []
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--capture", type=Path, required=True, help="RecorderManager motion NPZ or clip NPZ")
  parser.add_argument("--description", type=Path, required=True, help="Ascento URDF, package directory, or ZIP archive")
  parser.add_argument("--output", type=Path, required=True, help="Output .blend scene")
  parser.add_argument(
    "--warehouse",
    type=Path,
    default=DEFAULT_WAREHOUSE_PATH,
    help=f"Sketchfab warehouse FBX, package directory, or ZIP (default: {DEFAULT_WAREHOUSE_PATH})",
  )
  parser.add_argument("--root-link", default=DEFAULT_ROOT_LINK, help=f"URDF frame matched to root_pos (default: {DEFAULT_ROOT_LINK})")
  parser.add_argument("--fps", type=float, default=None, help="Override capture or clip frame rate")
  parser.add_argument("--render-dir", type=Path, default=None, help="Optional directory for rendered PNG frames")
  parser.add_argument(
    "--camera-shots",
    nargs="+",
    choices=CINEMATIC_SHOTS,
    metavar="SHOT",
    help="Ordered cinematic cuts: low_front, side_follow, rear_chase, orbit",
  )
  parser.add_argument(
    "--cinematic-dof",
    action="store_true",
    help="Opt into depth of field, focused on the animated robot bounds",
  )
  parser.add_argument("--resolution", type=int, default=720, help="Render width in pixels (default: 720; cinematic shots use 16:9)")
  args = parser.parse_args(arguments)
  if args.resolution < 16:
    parser.error("--resolution must be at least 16 pixels")
  if args.cinematic_dof and not args.camera_shots:
    parser.error("--cinematic-dof requires --camera-shots")
  if args.output.suffix.lower() != ".blend":
    parser.error("--output must use the .blend extension")
  return args


if __name__ == "__main__":
  try:
    run(_parse_args())
  except Exception as error:
    print(f"ERROR: {error}", file=sys.stderr)
    raise
