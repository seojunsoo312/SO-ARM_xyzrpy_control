"""Project base frame vs Pinocchio/URDF world.

User-facing base (pendant, teach place, hand-eye, register --base) is the
URDF world rotated by Rz(180°) about the shared origin. +Z is unchanged.

  p_urdf = R @ p_user
  R_urdf = R @ R_user
  T_urdf = T_urdf_from_user @ T_user

with R = Rz(180°) = diag(-1, -1, 1). Internal FK/IK/collision stay in URDF.
"""

from __future__ import annotations

import numpy as np

from motion.robot_kinematics import TcpPose, rotmat_to_rpy_deg, rpy_deg_to_rotmat

# p_urdf = R_URDF_FROM_USER @ p_user
R_URDF_FROM_USER = np.diag([-1.0, -1.0, 1.0])


def T_urdf_from_user_matrix() -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R_URDF_FROM_USER
    return T


def xyz_urdf_from_user(xyz: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(xyz, dtype=float).reshape(3)


def xyz_user_from_urdf(xyz: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(xyz, dtype=float).reshape(3)


def rot_urdf_from_user(R_user: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(R_user, dtype=float).reshape(3, 3)


def rot_user_from_urdf(R_urdf: np.ndarray) -> np.ndarray:
    return R_URDF_FROM_USER @ np.asarray(R_urdf, dtype=float).reshape(3, 3)


def T_urdf_from_user(T_user: np.ndarray) -> np.ndarray:
    return T_urdf_from_user_matrix() @ np.asarray(T_user, dtype=float).reshape(4, 4)


def T_user_from_urdf(T_urdf: np.ndarray) -> np.ndarray:
    return T_urdf_from_user_matrix() @ np.asarray(T_urdf, dtype=float).reshape(4, 4)


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
