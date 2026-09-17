"""로컬 윗면 PCA. 화면 디버그·등록 실패 폴백. 로봇 명령의 정본이 아니다.

정본 xyzrpy 는 yolo.pose.register (CAD vs ROI).
"""

from __future__ import annotations

import numpy as np

from yolo.pose.depth_cloud import pca_obb
from yolo.pose.instances import height_above_plane

SLICE_MM = 6.0
MIN_SLICE_POINTS = 20


def top_slice(
    xyz: np.ndarray,
    heights: np.ndarray,
    band_mm: float = SLICE_MM,
) -> np.ndarray:
    """Keep points near the 95th-percentile height (side walls drop out)."""
    h = np.asarray(heights, dtype=np.float64).reshape(-1)
    if len(h) == 0:
        return np.zeros(len(xyz), dtype=bool)
    h_top = float(np.percentile(h, 95))
    return h >= (h_top - abs(float(band_mm)))


def project_to_plane(xyz: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """3D points → 2D coords in a plane frame (origin = centroid, z = normal)."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    n = np.asarray(plane[:3], dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    if abs(n[2]) < 0.9:
        x_axis = np.cross(np.array([0.0, 0.0, 1.0]), n)
    else:
        x_axis = np.cross(np.array([0.0, 1.0, 0.0]), n)
    x_axis = x_axis / max(float(np.linalg.norm(x_axis)), 1e-12)
    y_axis = np.cross(n, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)
    origin = pts.mean(axis=0) if len(pts) else np.zeros(3)
    rel = pts - origin
    return np.column_stack((rel @ x_axis, rel @ y_axis))


def pca_xy(xy: np.ndarray) -> dict | None:
    """2D PCA. yaw_deg in [-90, 90), 180° period."""
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) < 3:
        return None
    center = np.median(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    values, axes = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order]
    axis = axes[:, 0]
    yaw = float(np.degrees(np.arctan2(axis[1], axis[0])))
    yaw = (yaw + 90.0) % 180.0 - 90.0
    return {
        "center": center.astype(np.float64),
        "axis": axis.astype(np.float64),
        "yaw_deg": yaw,
        "eigenvalues": values.astype(np.float64),
    }


def fit_local_plane(xyz: np.ndarray) -> np.ndarray | None:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None
    center = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
    normal = vh[-1]
    normal = normal / max(float(np.linalg.norm(normal)), 1e-12)
    d = -float(np.dot(normal, center))
    if d > 0:
        normal = -normal
        d = -d
    return np.r_[normal, d]


def local_pose(
    xyz: np.ndarray,
    *,
    desk_plane: np.ndarray | None = None,
    band_mm: float = SLICE_MM,
    in_base: bool = False,
) -> dict | None:
    """Top-slice → local plane → yaw on that plane. xyz already in working frame."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < MIN_SLICE_POINTS:
        return None
    if in_base:
        heights = pts[:, 2].copy()
    elif desk_plane is not None:
        heights = height_above_plane(pts, desk_plane)
    else:
        heights = -pts[:, 2]
    keep = top_slice(pts, heights, band_mm=band_mm)
    sliced = pts[keep]
    if len(sliced) < 8:
        sliced = pts
    local = fit_local_plane(sliced)
    if local is None:
        return None
    xy = project_to_plane(sliced, local)
    pca = pca_xy(xy)
    if pca is None:
        return None
    center = np.median(sliced, axis=0)
    normal = local[:3]
    if in_base and normal[2] < 0:
        normal = -normal
    return {
        "center_mm": center.astype(np.float64),
        "normal": normal.astype(np.float64),
        "yaw_deg": float(pca["yaw_deg"]),
        "z_top": float(np.percentile(heights[keep] if np.any(keep) else heights, 95)),
        "n_points": int(len(sliced)),
        "obb": pca_obb(sliced.astype(np.float32)),
        "sliced_xyz": sliced.astype(np.float32),
        "source": "local_plane",
        "axis_2d": pca["axis"],
    }
