"""Instance ROI clouds, top-slice, table-plane PCA pose (base mm if T given).

Does not open the camera, YOLO, or the serial bus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.transforms import apply_T
from yolo.depth_cloud import (
    Z_MAX_MM,
    Z_MIN_MM,
    largest_component,
    mask_from_seg,
    mask_from_seg_xy,
    mask_from_xyxy,
    pca_obb,
    plane_foreground_mask,
    points_from_mask,
)

SLICE_MM = 6.0
MIN_SLICE_POINTS = 20


@dataclass
class InstanceCloud:
    xyxy: list[float]
    roi: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    height_score: float
    n_zok: int = 0


def height_above_plane(xyz: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Desk plane ax+by+cz+d=0. Camera-side objects have negative signed distance."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    signed = pts @ np.asarray(plane[:3], dtype=np.float64) + float(plane[3])
    return -signed


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


def _height_score(xyz: np.ndarray, plane: np.ndarray | None, *, in_base: bool) -> float:
    if len(xyz) == 0:
        return float("-inf")
    if in_base:
        return float(np.percentile(xyz[:, 2], 95))
    if plane is not None:
        return float(np.percentile(height_above_plane(xyz, plane), 95))
    return float(-np.median(xyz[:, 2]))


def collect_instances(
    *,
    depth,
    bgr,
    K: np.ndarray,
    result,
    plane: np.ndarray | None,
    use_mask: bool,
    pad: int,
    plane_mm: float,
    stride: int,
    T_base_cam: np.ndarray | None,
) -> list[InstanceCloud]:
    """One cloud per YOLO box (optional seg ∩ desk foreground)."""
    h, w = bgr.shape[:2]
    out: list[InstanceCloud] = []
    if result.boxes is None:
        return out
    in_base = T_base_cam is not None
    for i, box in enumerate(result.boxes):
        xyxy = box.xyxy[0].tolist()
        roi = mask_from_xyxy(h, w, xyxy, pad=pad)
        if use_mask and result.masks is not None:
            if result.masks.xy is not None and i < len(result.masks.xy):
                roi = roi & mask_from_seg_xy(result.masks.xy, i, h, w)
            elif result.masks.data is not None and i < len(result.masks.data):
                roi = roi & mask_from_seg(result.masks.data, i, h, w)
        n_zok = int(
            np.count_nonzero(
                roi & (depth > Z_MIN_MM) & (depth < Z_MAX_MM)
            )
        )
        # 박스만 쓸 때는 책상 4mm 제거. 마스크는 이미 물체라서 얇은 브래킷까지 지우지 않는다.
        if plane is not None and not use_mask:
            roi = roi & plane_foreground_mask(depth, K, plane, height_mm=plane_mm)
        roi = largest_component(roi)
        xyz, rgb = points_from_mask(depth, K, roi, bgr=bgr, stride=stride)
        if in_base and len(xyz):
            xyz = apply_T(xyz, T_base_cam)
        score = _height_score(xyz, None if in_base else plane, in_base=in_base)
        out.append(
            InstanceCloud(
                xyxy=xyxy,
                roi=roi,
                xyz=xyz,
                rgb=rgb,
                height_score=score,
                n_zok=n_zok,
            )
        )
    return out


def select_topmost(instances: list[InstanceCloud]) -> int | None:
    scored = [
        (i, inst.height_score, len(inst.xyz))
        for i, inst in enumerate(instances)
        if len(inst.xyz)
    ]
    if not scored:
        return None
    scored.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return int(scored[0][0])


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
