import os
import numpy as np
from unitree_rl_gym.resources.task_assets.playboard.odio_urdf import *
import numpy
from pint import UnitRegistry

ureg = UnitRegistry()

wood = Material(Color(rgba="0.9 0.6 0.3 1"))
red = Material(Color(rgba="1 0 0 1"))
black = Material(Color(rgba="0.2 0.2 0.2 1"))
blue = Material(Color(rgba="0 0 1 1"))
white = Material(Color(rgba="1 1 1 1"))
grey = Material(Color(rgba="0.7 0.7 0.7 1"))

wall_length = (18 * ureg.inch).to(ureg.meter).magnitude
wall_height = (24 * ureg.inch).to(ureg.meter).magnitude
knob_holder_thickness = (0.015 * ureg.meter).magnitude
knob_1_to_wall_bottom_line_dist = (14.406 * ureg.inch).to(ureg.meter).magnitude
knob_1_to_wall_side_line_dist = (3.135 * ureg.inch).to(ureg.meter).magnitude
knob_to_knob_center_dist = (3.937 * ureg.inch).to(ureg.meter).magnitude

mm2meterscale = "0.001 0.001 0.001"

wall_link = Link(
    Visual(
        Origin(xyz=[0, wall_length, 0], rpy=[0, 0, np.pi]),
        Geometry(Mesh(filename="meshes/wall.stl", scale=mm2meterscale)),
        wood,
    ),
    Collision(
        Origin(xyz=[0, wall_length, 0], rpy=[0, 0, np.pi]),
        Geometry(Mesh(filename="meshes/wall.stl", scale=mm2meterscale)),
    ),
    Inertial(
        Origin(xyz=[0, wall_length, 0], rpy=[0, 0, np.pi]),
        Mass(value=100),
        Inertia(ixx=1.0, ixy=0.0, ixz=0.0, iyy=1.0, iyz=0.0, izz=1.0),
    ),
    name="wall_link",
)




knob_holder_1 = Link(
    Visual(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_1.stl", scale=mm2meterscale)),
        grey,
    ),
    Collision(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_1.stl", scale=mm2meterscale)),
        grey,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.2),
        # TODO: calculate inertia for real
        Inertia(ixx=0.000833, ixy=0, ixz=0, iyy=0.000833, iyz=0, izz=0.000833),
    ),
    name="knob_holder_1",
)
knob_holder_1_joint = Joint(
    Parent(wall_link),
    Child(knob_holder_1),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=-0.5 * np.pi, upper=np.pi),
    Dynamics(friction=0.01, damping=0),
    type="fixed",
    name=f"knob_holder_1_joint",
)
knob_1 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_1.stl", scale=mm2meterscale)),
        black,
    ),
    Collision(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_1.stl", scale=mm2meterscale)),
        white,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.4),
        Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
    ),
    name="knob_1",
)
knob_1_joint = Joint(
    Parent(knob_holder_1),
    Child(knob_1),
    Origin(xyz=[knob_holder_thickness + 0.026, knob_1_to_wall_side_line_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=0, upper=2 * np.pi),
    Dynamics(friction=0.01, damping=0),
    type="revolute",
    name=f"knob_1_joint",
)
knob_indicator_1 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Box(size=[0.01, 0.0025, 0.001])),
        red,
    ),
    name="knob_indicator_1",
)
knob_indicator_1_joint_rigid = Joint(
    Parent(knob_1),
    Child(knob_indicator_1),
    Origin(xyz=[0.005, 0, 0.006], rpy=[0, 0, 0]),
    type="fixed",
    name="knob_indicator_1_joint_rigid",
)






knob_holder_2 = Link(
    Visual(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist + knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_2.stl", scale=mm2meterscale)),
        grey,
    ),
    Collision(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist + knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_2.stl", scale=mm2meterscale)),
        grey,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.2),
        # TODO: calculate inertia for real
        Inertia(ixx=0.000833, ixy=0, ixz=0, iyy=0.000833, iyz=0, izz=0.000833),
    ),
    name="knob_holder_2",
)
knob_holder_2_joint = Joint(
    Parent(wall_link),
    Child(knob_holder_2),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=-0.5 * np.pi, upper=np.pi),
    Dynamics(friction=0.01, damping=0),
    type="fixed",
    name=f"knob_holder_2_joint",
)
knob_2 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_2.stl", scale=mm2meterscale)),
        black,
    ),
    Collision(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_2.stl", scale=mm2meterscale)),
        white,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.4),
        Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
    ),
    name="knob_2",
)
knob_2_joint = Joint(
    Parent(knob_holder_2),
    Child(knob_2),
    Origin(xyz=[knob_holder_thickness + 0.012, knob_1_to_wall_side_line_dist + knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=0, upper=2 * np.pi),
    Dynamics(friction=0.01, damping=0),
    type="revolute",
    name=f"knob_2_joint",
)
knob_indicator_2 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[np.pi / 2, np.pi, np.pi / 2]),
        Geometry(Box(size=[0.01, 0.0025, 0.001])),
        red,
    ),
    name="knob_indicator_2",
)
knob_indicator_2_joint_rigid = Joint(
    Parent(knob_2),
    Child(knob_indicator_2),
    Origin(xyz=[0.01, 0.012, 0], rpy=[0, 0, 0]),
    type="fixed",
    name="knob_indicator_2_joint_rigid",
)
























knob_holder_3 = Link(
    Visual(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist + 2 * knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_3.stl", scale=mm2meterscale)),
        grey,
    ),
    Collision(
        Origin(xyz=[knob_holder_thickness, knob_1_to_wall_side_line_dist + 2 * knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, - np.pi / 2]),
        Geometry(Mesh(filename="meshes/knob_holder_3.stl", scale=mm2meterscale)),
        grey,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.2),
        # TODO: calculate inertia for real
        Inertia(ixx=0.000833, ixy=0, ixz=0, iyy=0.000833, iyz=0, izz=0.000833),
    ),
    name="knob_holder_3",
)
knob_holder_3_joint = Joint(
    Parent(wall_link),
    Child(knob_holder_3),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=-0.5 * np.pi, upper=np.pi),
    Dynamics(friction=0.01, damping=0),
    type="fixed",
    name=f"knob_holder_3_joint",
)
knob_3 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_3.stl", scale=mm2meterscale)),
        black,
    ),
    Collision(
        Origin(xyz=[0, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Mesh(filename="meshes/knob_3.stl", scale=mm2meterscale)),
        white,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.4),
        Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
    ),
    name="knob_3",
)
knob_3_joint = Joint(
    Parent(knob_holder_3),
    Child(knob_3),
    Origin(xyz=[knob_holder_thickness + 0.012, knob_1_to_wall_side_line_dist + 2 * knob_to_knob_center_dist, knob_1_to_wall_bottom_line_dist], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=0, upper=2 * np.pi),
    Dynamics(friction=0.01, damping=0),
    type="revolute",
    name=f"knob_3_joint",
)
knob_indicator_3 = Link(
    Visual(
        Origin(xyz=[0, 0, 0], rpy=[np.pi / 2, np.pi, np.pi / 2]),
        Geometry(Box(size=[0.010, 0.0025, 0.001])),
        red,
    ),
    name="knob_indicator_3",
)
knob_indicator_3_joint_rigid = Joint(
    Parent(knob_3),
    Child(knob_indicator_3),
    Origin(xyz=[0.019, 0.016, 0], rpy=[0, 0, 0]),
    type="fixed",
    name="knob_indicator_3_joint_rigid",
)






knob_set_1 = Group(knob_holder_1, knob_holder_1_joint, knob_1, knob_1_joint, knob_indicator_1, knob_indicator_1_joint_rigid)
knob_set_2 = Group(knob_holder_2, knob_holder_2_joint, knob_2, knob_2_joint, knob_indicator_2, knob_indicator_2_joint_rigid)
knob_set_3 = Group(knob_holder_3, knob_holder_3_joint, knob_3, knob_3_joint, knob_indicator_3, knob_indicator_3_joint_rigid)
playboard = Group(wall_link, knob_set_1, knob_set_2, knob_set_3)

robot = Robot(playboard, name="knob_exp")

save_path = os.path.abspath(
    f"./playboard.urdf"
)

# Save the URDF to a file (optional)
with open(save_path, "w") as f:
    f.write(robot.urdf())
