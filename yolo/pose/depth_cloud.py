"""Aligned RGB-D → camera-frame point cloud inside a 2D ROI."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from vision.calib import load_K, load_T_base_cam
from vision.rgbd import OrbbecV1

Z_MIN_MM = 80.0
Z_MAX_MM = 1800.0


def open_orbbec(width: int = 640, height: int = 480, **kwargs):
    return OrbbecV1(width, height, **kwargs)


def rotate180(bgr, depth_mm):
    if bgr is not None:
        bgr = cv2.rotate(bgr, cv2.ROTATE_180)
    if depth_mm is not None:
        depth_mm = cv2.rotate(depth_mm, cv2.ROTATE_180)
    return bgr, depth_mm


def colorize_depth(
    depth_mm: np.ndarray, center_span_mm: float = 60.0
) -> np.ndarray:
    """깊이를 보기 좋게 자동 확대한다. 점군 계산에는 영향을 주지 않는다."""
    vis = np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    valid = (depth_mm > Z_MIN_MM) & (depth_mm < Z_MAX_MM)
    if not np.any(valid):
        return vis

    values = depth_mm[valid]
    center = float(np.median(values))
    # 작업대 깊이 주변을 확대한다. 극단값과 프레임 가장자리 노이즈는 무시한다.
    low = max(Z_MIN_MM, center - center_span_mm / 2.0)
    high = min(Z_MAX_MM, center + center_span_mm / 2.0)
    norm = np.clip(depth_mm, low, high)
    norm = ((norm - low) / max(high - low, 1.0) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    vis[valid] = color[valid]
    cv2.putText(
        vis,
        f"depth {low:.0f}-{high:.0f} mm",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return vis


def mask_from_quad(
    h: int, w: int, quad, pad: int = 2
) -> np.ndarray:
    """OBB 네 꼭짓점으로 ROI. pad 는 팽창 픽셀."""
    mask = np.zeros((h, w), dtype=np.uint8)
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    if pad > 0 and np.any(mask):
        k = 2 * int(pad) + 1
        mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
    return mask.astype(bool)


def mask_from_xyxy(
    h: int, w: int, xyxy: list[float], pad: int = 2
) -> np.ndarray:
    x1, y1, x2, y2 = xyxy
    xa = max(0, int(np.floor(min(x1, x2))) - pad)
    xb = min(w, int(np.ceil(max(x1, x2))) + pad)
    ya = max(0, int(np.floor(min(y1, y2))) - pad)
    yb = min(h, int(np.ceil(max(y1, y2))) + pad)
    mask = np.zeros((h, w), dtype=bool)
    if xb > xa and yb > ya:
        mask[ya:yb, xa:xb] = True
    return mask


def mask_from_seg(masks_data, index: int, h: int, w: int) -> np.ndarray:
    raw = masks_data[index]
    if hasattr(raw, "cpu"):
        raw = raw.cpu().numpy()
    raw = np.asarray(raw)
    if raw.ndim == 3:
        raw = raw[0]
    resized = cv2.resize(raw.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    return resized > 0.5


def mask_from_seg_xy(masks_xy, index: int, h: int, w: int) -> np.ndarray:
    """원본 해상도 폴리곤. letterbox된 masks.data 리사이즈보다 맞다."""
    mask = np.zeros((h, w), dtype=np.uint8)
    if masks_xy is None or index >= len(masks_xy):
        return mask.astype(bool)
    pts = np.asarray(masks_xy[index], dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        return mask.astype(bool)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
    return mask.astype(bool)


def fit_plane_ransac(
    xyz: np.ndarray,
    threshold_mm: float = 3.0,
    iterations: int = 200,
    min_abs_nz: float = 0.7,
) -> tuple[np.ndarray, int]:
    """가장 큰 전방 평면 ax+by+cz+d=0을 찾는다."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 100:
        raise ValueError("평면 추정 점이 부족합니다.")
    if len(pts) > 12000:
        rng = np.random.default_rng(42)
        pts = pts[rng.choice(len(pts), 12000, replace=False)]
    rng = np.random.default_rng(42)
    best_plane = None
    best_inliers = None
    best_count = 0
    for _ in range(iterations):
        p0, p1, p2 = pts[rng.choice(len(pts), 3, replace=False)]
        normal = np.cross(p1 - p0, p2 - p0)
        length = float(np.linalg.norm(normal))
        if length < 1e-6:
            continue
        normal /= length
        if abs(float(normal[2])) < min_abs_nz:
            continue
        d = -float(np.dot(normal, p0))
        distances = np.abs(pts @ normal + d)
        inliers = distances <= threshold_mm
        count = int(np.count_nonzero(inliers))
        if count > best_count:
            best_count = count
            best_plane = np.r_[normal, d]
            best_inliers = inliers
    if best_plane is None or best_inliers is None or best_count < 100:
        raise ValueError("책상 평면을 찾지 못했습니다.")

    # RANSAC inlier 전체로 평면을 다시 맞춘다.
    fit = pts[best_inliers]
    center = fit.mean(axis=0)
    _, _, vh = np.linalg.svd(fit - center, full_matrices=False)
    normal = vh[-1]
    normal /= np.linalg.norm(normal)
    d = -float(np.dot(normal, center))
    # 카메라 원점이 음의 쪽, 책상 위 물체도 음의 쪽이 되도록 통일한다.
    if d > 0:
        normal = -normal
        d = -d
    plane = np.r_[normal, d].astype(np.float64)
    count = int(np.count_nonzero(np.abs(pts @ normal + d) <= threshold_mm))
    return plane, count


def plane_foreground_mask(
    depth_mm: np.ndarray,
    K: np.ndarray,
    plane: np.ndarray,
    height_mm: float = 4.0,
) -> np.ndarray:
    """책상 평면보다 카메라 쪽으로 height_mm 이상 돌출된 픽셀만 남긴다."""
    h, w = depth_mm.shape[:2]
    vv, uu = np.indices((h, w), dtype=np.float64)
    z = depth_mm.astype(np.float64)
    x = (uu - float(K[0, 2])) * z / float(K[0, 0])
    y = (vv - float(K[1, 2])) * z / float(K[1, 1])
    signed = plane[0] * x + plane[1] * y + plane[2] * z + plane[3]
    valid = (z > Z_MIN_MM) & (z < Z_MAX_MM)
    return valid & (signed < -abs(height_mm))


def largest_component(mask: np.ndarray, min_area: int = 20) -> np.ndarray:
    """고립된 뎁스 노이즈를 버리고 가장 큰 연결 성분만 남긴다."""
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
    if count <= 1:
        return np.zeros_like(mask, dtype=bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    index = int(np.argmax(areas)) + 1
    if int(stats[index, cv2.CC_STAT_AREA]) < min_area:
        return np.zeros_like(mask, dtype=bool)
    return labels == index


def pca_obb(xyz: np.ndarray) -> dict[str, np.ndarray] | None:
    """점군의 PCA 중심, 주축, 축 방향 크기를 계산한다."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) < 3:
        return None
    center = np.median(pts, axis=0)
    centered = pts - center
    cov = np.cov(centered, rowvar=False)
    values, axes = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values = values[order]
    axes = axes[:, order]
    projected = centered @ axes
    low = np.percentile(projected, 2, axis=0)
    high = np.percentile(projected, 98, axis=0)
    return {
        "center": center.astype(np.float32),
        "axes": axes.astype(np.float32),
        "extents": (high - low).astype(np.float32),
        "eigenvalues": values.astype(np.float32),
    }


def points_from_mask(
    depth_mm: np.ndarray,
    K: np.ndarray,
    mask: np.ndarray,
    bgr=None,
    stride: int = 2,
    z_min: float = Z_MIN_MM,
    z_max: float = Z_MAX_MM,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = depth_mm.shape[:2]
    us = np.arange(0, w, stride)
    vs = np.arange(0, h, stride)
    uu, vv = np.meshgrid(us, vs)
    keep = mask[vv, uu]
    z = depth_mm[vv, uu]
    keep = keep & (z > z_min) & (z < z_max)
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    u = uu[keep].astype(np.float64)
    v = vv[keep].astype(np.float64)
    z = z[keep].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    if bgr is None:
        rgb = np.zeros((len(xyz), 3), dtype=np.uint8)
    else:
        rgb = bgr[vv[keep], uu[keep]][:, ::-1].copy()
    return xyz, rgb


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if rgb is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")
        else:
            for p, c in zip(xyz, rgb):
                f.write(
                    f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f} {int(c[0])} {int(c[1])} {int(c[2])}\n"
                )
