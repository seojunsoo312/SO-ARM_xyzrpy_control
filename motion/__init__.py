"""Arm motion: IK, serial bus, jog/go-to loop, Meshcat view. No YOLO.

URDF and meshes live in motion/robot/. Meshcat view is motion/visualizer.py.
Follower calibration is the lerobot-calibrate cache on this PC.
교육용 Arm 은 Meshcat 뷰어를 연다.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

from motion.controller import (
    GOTO_DURATION_S,
    GRIPPER_VEL_UNIT_S,
    JOG_VEL_MPS,
    JOINT_VEL_DEG_S,
    Controller,
    PendantState,
)
from motion.hw_controller import (
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    Hardware,
    grip_100_to_user,
    grip_user_to_100,
)
from motion.arm import Arm, ArmError
from motion.pick_targets import PickTargets, load_pick_targets
from motion.pose_server import POSE_URL, PoseServer, fetch_pose
from motion.robot_kinematics import (
    DEFAULT_URDF,
    EE_FRAME,
    ROBOT_DIR,
    TCP_FRAME,
    TCP_OFFSET_IN_L6,
    URDF_JOINT_NAMES,
    RobotKinematics,
    TcpPose,
)

__all__ = [
    "DEFAULT_PORT",
    "DEFAULT_ROBOT_ID",
    "DEFAULT_URDF",
    "EE_FRAME",
    "GOTO_DURATION_S",
    "GRIPPER_VEL_UNIT_S",
    "Hardware",
    "JOG_VEL_MPS",
    "JOINT_VEL_DEG_S",
    "POSE_URL",
    "PendantState",
    "PoseServer",
    "PROJECT_ROOT",
    "ROBOT_DIR",
    "RobotKinematics",
    "TCP_FRAME",
    "TCP_OFFSET_IN_L6",
    "TcpPose",
    "URDF_JOINT_NAMES",
    "Arm",
    "ArmError",
    "PickTargets",
    "load_pick_targets",
    "Controller",
    "fetch_pose",
    "grip_100_to_user",
    "grip_user_to_100",
]
