"""Robot-agnostic 4x4 rigid transforms and camera unprojection (mm)."""

from __future__ import annotations

import numpy as np


def rt_to_T(R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=float).reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float)
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def apply_T(xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    if len(xyz) == 0:
        return xyz
    ones = np.ones((len(xyz), 1), dtype=np.float64)
    homo = np.hstack([xyz.astype(np.float64), ones])
    out = (T @ homo.T).T
    return out[:, :3].astype(np.float32)


def unproject(u, v, z_mm, K) -> np.ndarray:
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    return np.array([(u - cx) * z_mm / fx, (v - cy) * z_mm / fy, z_mm], dtype=float)
