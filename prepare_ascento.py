"""Create a MuJoCo Playground-ready Guard 2.0-oriented physics model.

Rigid-body masses/inertias are retained from the supplied Ascento description.
This patch adds finite leg travel, wheel-specific tyre contact, and hard actuator
force ranges. Dynamic torque-speed/bandwidth behavior lives in guard2_physics.py.
Guard 2.0 has no passive physical leg spring; none is added here.
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
from guard2_physics import (
    JOINT_NAMES, LEG_INDEX, WHEEL_INDEX,
    LEG_Q_MIN, LEG_Q_MAX,
    LEG_ACTUATOR, WHEEL_ACTUATOR,
    TIRE_FRICTION, TIRE_CONDIM, TIRE_SOLREF, TIRE_SOLIMP,
)

ROOT = Path(__file__).resolve().parent
source = ROOT / 'ascento.urdf'
physics = ROOT / 'ascento_physics.urdf'

tree = ET.parse(source)
robot = tree.getroot()
robot.insert(0, ET.Comment(
    'Guard 2.0-oriented MuJoCo training copy. CAD masses/inertias retained. '
    'Finite leg travel and tyre contact added. No passive leg spring.'
))
for link in robot.findall('link'):
    for visual in list(link.findall('visual')):
        link.remove(visual)
    for collision in list(link.findall('collision')):
        if collision.find('.//mesh') is not None:
            link.remove(collision)
for tag in ('ros2_control', 'gazebo'):
    for node in list(robot.findall(tag)):
        robot.remove(node)
compiler = robot.find('./mujoco/compiler')
if compiler is not None:
    compiler.set('meshdir', '.')
    compiler.set('discardvisual', 'true')
ET.indent(tree, space='  ')
tree.write(physics, encoding='utf-8', xml_declaration=True)

spec = mujoco.MjSpec.from_file(str(physics))
base = next(b for b in spec.bodies if b.name == 'base_link')
base.name = 'base'
body_names = {
    'ascento/thigh_left': 'left_thigh',
    'ascento/shank_left': 'left_shank',
    'ascento/wheel_left': 'left_wheel',
    'ascento/thigh_right': 'right_thigh',
    'ascento/shank_right': 'right_shank',
    'ascento/wheel_right': 'right_wheel',
}
for body in spec.bodies:
    if body.name in body_names:
        body.name = body_names[body.name]
joint_names = {
    'ascento/hip_left': 'left_hip',
    'ascento/knee_left': 'left_knee',
    'ascento/ankle_left': 'left_wheel_joint',
    'ascento/hip_right': 'right_hip',
    'ascento/knee_right': 'right_knee',
    'ascento/ankle_right': 'right_wheel_joint',
}
for joint in spec.joints:
    if joint.name in joint_names:
        joint.name = joint_names[joint.name]
base.add_freejoint()
imu = base.add_site()
imu.name = 'imu_reference'
imu.pos = [0.0, 0.0, 0.0]

# Collision/contact defaults. Internal self-collision stays disabled for MJX;
# floor contact remains active. Wheel cylinders receive compliant tyre settings.
for geom in spec.geoms:
    geom.contype = 1
    geom.conaffinity = 2
    geom.rgba = [.18, .30, .48, 1]
    try:
        is_tire = geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER and float(geom.size[0]) >= 0.20
    except Exception:
        is_tire = False
    if is_tire:
        geom.friction = list(TIRE_FRICTION)
        geom.condim = TIRE_CONDIM
        geom.solref = list(TIRE_SOLREF)
        geom.solimp = list(TIRE_SOLIMP)

# Enforce finite leg joint limits in MuJoCo as a second line of defense.
for joint in spec.joints:
    if joint.name in (JOINT_NAMES[i] for i in LEG_INDEX):
        joint.limited = True
        joint.range = [LEG_Q_MIN, LEG_Q_MAX]
        # No passive stiffness: Guard 2.0 has no physical leg spring.
        joint.stiffness = [0.0, 0.0, 0.0]

floor = spec.worldbody.add_geom()
floor.name = 'floor'
floor.type = mujoco.mjtGeom.mjGEOM_PLANE
floor.size = [20, 20, 0.1]
floor.friction = list(TIRE_FRICTION)
floor.solref = list(TIRE_SOLREF)
floor.solimp = list(TIRE_SOLIMP)
floor.contype = 2
floor.conaffinity = 1
floor.rgba = [.12, .13, .16, 1]

light = spec.worldbody.add_light()
light.name = 'studio_key'
light.pos = [-1.5, -2.5, 3]
light.diffuse = [1, 1, 1]
light.ambient = [.90, .90, .90]

fill = spec.worldbody.add_light()
fill.name = 'studio_fill'
fill.pos = [1.5, 1.5, 2.2]
fill.diffuse = [.90, .90, .90]
fill.ambient = [.30, .30, .30]

front = spec.worldbody.add_light()
front.name = 'studio_front'
front.pos = [0, -3, 1.2]
front.diffuse = [1, 1, 1]
front.ambient = [.40, .40, .40]

# Raw MuJoCo actuators retain only absolute hard force saturation.  The separate
# Guard2ActuatorModel applies torque-loop bandwidth, speed limits, and torque-speed
# derating at every simulation substep. No artificial command transport delay is used.
for i, name in enumerate(JOINT_NAMES):
    profile = WHEEL_ACTUATOR if i in WHEEL_INDEX else LEG_ACTUATOR
    actuator = spec.add_actuator()
    actuator.name = name.replace('/', '_') + '_motor'
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.target = name
    actuator.set_to_motor()
    actuator.ctrllimited = True
    actuator.ctrlrange = [-profile.peak_torque_nm, profile.peak_torque_nm]
    try:
        actuator.forcelimited = True
        actuator.forcerange = [-profile.peak_torque_nm, profile.peak_torque_nm]
    except AttributeError:
        pass

model = spec.compile()
model_dir = ROOT / 'model'
model_dir.mkdir(exist_ok=True)
static_xml = model_dir / 'ascento_guard2_mjx.xml'
static_xml.write_text(spec.to_xml(), encoding='utf-8')
print('PLAYGROUND_IMPORT_OK')
print('nq=', model.nq, ' nv=', model.nv, ' nu=', model.nu)
print('static_xml=', static_xml)
print('actuators=', [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
print('output=', physics)
