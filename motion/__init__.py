"""Arm motion: IK, serial bus, jog/go-to loop. No YOLO.

Assets that stay in pendant/: URDF, meshes, Feetech calibration JSON.
교육용 Arm 은 Meshcat 뷰어를 연다.
"""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
PENDANT_DIR = PROJECT_ROOT / "pendant"
DEFAULT_URDF = PENDANT_DIR / "SO101_6DOF.urdf"
BUNDLED_CALIBRATION_DIR = PENDANT_DIR / "calibration" / "so_follower"

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
from motion.pose_server import POSE_URL, PoseServer, fetch_pose
from motion.robot_kinematics import (
    EE_FRAME,
    TCP_FRAME,
    TCP_OFFSET_IN_L6,
    URDF_JOINT_NAMES,
    RobotKinematics,
    TcpPose,
)

__all__ = [
    "BUNDLED_CALIBRATION_DIR",
    "DEFAULT_PORT",
    "DEFAULT_ROBOT_ID",
    "DEFAULT_URDF",
    "EE_FRAME",
    "GOTO_DURATION_S",
    "GRIPPER_VEL_UNIT_S",
    "Hardware",
    "JOG_VEL_MPS",
    "JOINT_VEL_DEG_S",
    "PENDANT_DIR",
    "POSE_URL",
    "PendantState",
    "PoseServer",
    "PROJECT_ROOT",
    "RobotKinematics",
    "TCP_FRAME",
    "TCP_OFFSET_IN_L6",
    "TcpPose",
    "URDF_JOINT_NAMES",
    "Arm",
    "ArmError",
    "Controller",
    "fetch_pose",
    "grip_100_to_user",
    "grip_user_to_100",
]
