import numpy as np
from ..utils.odio_urdf import *
import os

wood = Material(Color(rgba="0.8039 0.6667 0.4902 1"))
red = Material(Color(rgba="1 0 0 1"))
black = Material(Color(rgba="0.2 0.2 0.2 1"))
blue = Material(Color(rgba="0 0 1 1"))
white = Material(Color(rgba="1 1 1 1"))
grey = Material(Color(rgba="0.7 0.7 0.7 1"))


base_link = Link(
    Visual(
        Origin(xyz=[-0.05, 0, 0], rpy=[0, 0, 0]),
        Geometry(Box(size=[0.1, 1, 1])),
        wood,
    ),
    Collision(
        Origin(xyz=[-0.05, 0, 0], rpy=[0, 0, 0]),
        Geometry(Box(size=[0.1, 1, 1])),
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=2.5),
        Inertia(ixx=1.0, ixy=0.0, ixz=0.0, iyy=1.0, iyz=0.0, izz=1.0),
    ),
    name="base_link",
)

knob_base = Link(
    Visual(
        Origin(xyz=[0.01, 0, 0], rpy=[0, 0, 0]),
        Geometry(Box(size=[0.02, 0.15, 0.15])),
        grey,
    ),
    Collision(
        Origin(xyz=[0.01, 0, 0], rpy=[0, 0, 0]),
        Geometry(Box(size=[0.02, 0.15, 0.15])),
        white,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.2),
        # TODO: calculate inertia for real
        Inertia(ixx=0.000833, ixy=0, ixz=0, iyy=0.000833, iyz=0, izz=0.000833),
    ),
    name="knob_base",
)

# knob_body = Link(
#     Visual(
#         Origin(xyz=[0.04, 0, 0], rpy=[0, np.pi / 2, 0]),
#         Geometry(Cylinder(radius=0.02, length=0.05)),
#         black,
#     ),
#     Visual(
#         Origin(xyz=[0.065, 0, 0.010], rpy=[0, np.pi / 2, 0]),
#         Geometry(Box(size=[0.015, 0.004, 0.003])),
#         red,
#     ),
#     Inertial(
#         Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
#         Mass(value=0.4),
#         Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
#     ),
#     Collision(
#         Origin(xyz=[0.04, 0, 0], rpy=[0, np.pi / 2, 0]),
#         Geometry(Cylinder(radius=0.02, length=0.05)),
#         white,
#     ),
#     name="knob",
# )
knob_body = Link(
    Visual(
        Origin(xyz=[0.04, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Cylinder(radius=0.02, length=0.05)),
        black,
    ),
    Inertial(
        Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
        Mass(value=0.4),
        Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
    ),
    Collision(
        Origin(xyz=[0.04, 0, 0], rpy=[0, np.pi / 2, 0]),
        Geometry(Cylinder(radius=0.02, length=0.05)),
        white,
    ),
    name="knob",
)
knob_indicator = Link(
    Visual(
        Origin(xyz=[0.065, 0, 0.010], rpy=[0, np.pi / 2, 0]),
        Geometry(Box(size=[0.015, 0.004, 0.003])),
        red,
    ),
    # Inertial(
    #     Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    #     Mass(value=0.4),
    #     Inertia(ixx=0.0001, ixy=0, ixz=0, iyy=0.0001, iyz=0, izz=0.0001),
    # ),
    # Collision(
    #     Origin(xyz=[0.065, 0, 0.010], rpy=[0, np.pi / 2, 0]),
    #     Geometry(Box(size=[0.015, 0.004, 0.003])),
    #     white,
    # ),
    name="indicator",
)


knob_base_joint_rigid = Joint(
    Parent(base_link),
    Child(knob_base),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    type="fixed",
    name="base_link_knob_base_joint",
)

knob_joint = Joint(
    Parent(knob_base),
    Child(knob_body),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    Axis(xyz=[1, 0, 0]),
    Limit(effort=2, velocity=5, lower=-0.5 * np.pi, upper=np.pi),
    Dynamics(friction=0.01, damping=0),
    type="revolute",
    name=f"actuator_{1}_joint",
)

knob_indicator_joint_rigid = Joint(
    Parent(knob_body),
    Child(knob_indicator),
    Origin(xyz=[0, 0, 0], rpy=[0, 0, 0]),
    type="fixed",
    name="knob_knob_indicator_joint",
)

knob = Group(knob_base, knob_body, knob_base_joint_rigid, knob_joint, knob_indicator, knob_indicator_joint_rigid)

group = Group(base_link, knob)


robot = Robot(group, name="knob_exp")

save_path = os.path.abspath(
    f"assets/urdf/FrankaKnob.urdf"
)

# Save the URDF to a file (optional)
with open(save_path, "w") as f:
    f.write(robot.urdf())
