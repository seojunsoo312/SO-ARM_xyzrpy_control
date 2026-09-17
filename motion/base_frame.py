"""Project base frame vs Pinocchio/URDF world.

User-facing base (pendant, teach place, hand-eye, register --base):
  +X = robot forward, +Z = up, +Y = left (right-hand).

That is the previous Rz(180°) user frame rotated by Rz(+90°) about the
shared origin. Equivalently, p_urdf = Rz(-90°) @ p_user.

  p_urdf = R @ p_user
  R_urdf = R @ R_user
  T_urdf = T_urdf_from_user @ T_user

Internal FK/IK/collision stay in URDF.
"""

from __future__ import annotations

import numpy as np

from motion.robot_kinematics import TcpPose, rotmat_to_rpy_deg, rpy_deg_to_rotmat

USER_BASE_FRAME = "project_xfwd"
LEGACY_BASE_FRAME = "project_rz180"

# p_urdf = R_URDF_FROM_USER @ p_user  (Rz(-90°))
R_URDF_FROM_USER = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
R_USER_FROM_URDF = R_URDF_FROM_USER.T

# Previous user (+X right, +Y forward) → current user (+X forward, +Y left).
R_USER_FROM_LEGACY = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)


def T_urdf_from_user_matrix() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_URDF_FROM_USER
    return T


def T_user_from_urdf_matrix() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_USER_FROM_URDF
    return T


def xyz_urdf_from_user(xyz: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(xyz, dtype=float).reshape(3)


def xyz_user_from_urdf(xyz: np.ndarray) -> np.ndarray:
    return R_USER_FROM_URDF @ np.asarray(xyz, dtype=float).reshape(3)


def rot_urdf_from_user(R_user: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(R_user, dtype=float).reshape(3, 3)


def rot_user_from_urdf(R_urdf: np.ndarray) -> np.ndarray:
    return R_USER_FROM_URDF @ np.asarray(R_urdf, dtype=float).reshape(3, 3)


def T_urdf_from_user(T_user: np.ndarray) -> np.ndarray:
    return T_urdf_from_user_matrix() @ np.asarray(T_user, dtype=float).reshape(4, 4)


def T_user_from_urdf(T_urdf: np.ndarray) -> np.ndarray:
    return T_user_from_urdf_matrix() @ np.asarray(T_urdf, dtype=float).reshape(4, 4)


def T_user_from_stored(T: np.ndarray, base_frame: str | None) -> np.ndarray:
    """Lift a stored 4×4 in `base_frame` into the current user base."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    frame = (base_frame or LEGACY_BASE_FRAME).strip()
    if frame == USER_BASE_FRAME:
        return T
    if frame == LEGACY_BASE_FRAME:
        out = np.eye(4)
        out[:3, :3] = R_USER_FROM_LEGACY
        return out @ T
    raise ValueError(f"unknown base_frame: {base_frame!r}")


def tcp_pose_user_from_urdf(pose: TcpPose) -> TcpPose:
    R_u = rot_user_from_urdf(pose.rotation)
    xyz_u = xyz_user_from_urdf(pose.xyz_m)
    return TcpPose(xyz_m=xyz_u, rpy_deg=rotmat_to_rpy_deg(R_u), rotation=R_u)


def ee_target_urdf_from_user(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """User Go-to (mm, deg) → URDF meters + rotation matrix."""
    xyz_u = np.asarray(xyz_mm, dtype=float).reshape(3) / 1000.0
    rpy = np.asarray(rpy_deg, dtype=float).reshape(3)
    R_u = rpy_deg_to_rotmat(float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return xyz_urdf_from_user(xyz_u), rot_urdf_from_user(R_u)
