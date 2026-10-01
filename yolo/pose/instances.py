"""YOLO 인스턴스마다 점군을 모으고 최상단을 고른다.

포즈(xyzrpy)는 여기 없다. 정본은 register.py (CAD). 로컬 면 PCA는 pose.debug.local_plane.
카메라·YOLO·시리얼은 열지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vision.transforms import apply_T
from yolo.pose.depth_cloud import (
    Z_MAX_MM,
    Z_MIN_MM,
    largest_component,
    mask_from_seg,
    mask_from_seg_xy,
    mask_from_xyxy,
    plane_foreground_mask,
    points_from_mask,
)

MASK_PLANE_MM = 3.0


@dataclass
class InstanceCloud:
    xyxy: list[float]
    roi: np.ndarray
    xyz: np.ndarray
    rgb: np.ndarray
    height_score: float
    n_zok: int = 0
    quad: np.ndarray | None = None  # (4, 2) 가로세로 박스 꼭짓점
    class_id: int | None = None


def height_above_plane(xyz: np.ndarray, plane: np.ndarray) -> np.ndarray:
    """Desk plane ax+by+cz+d=0. Camera-side objects have negative signed distance."""
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        return np.zeros(0, dtype=np.float64)
    signed = pts @ np.asarray(plane[:3], dtype=np.float64) + float(plane[3])
    return -signed


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
    obb = getattr(result, "obb", None)
    boxes = getattr(result, "boxes", None)
    dets = boxes if boxes is not None and len(boxes) else obb
    if dets is None:
        return out
    in_base = T_base_cam is not None
    for i, det in enumerate(dets):
        raw = det.xyxy[0]
        if hasattr(raw, "cpu"):
            raw = raw.cpu().numpy()
        xyxy = [float(v) for v in np.asarray(raw).reshape(-1)[:4]]
        xa, ya, xb, yb = xyxy
        quad = np.array([[xa, ya], [xb, ya], [xb, yb], [xa, yb]], dtype=np.float32)
        class_id = None
        cls = getattr(det, "cls", None)
        if cls is not None:
            if hasattr(cls, "cpu"):
                cls = cls.cpu().numpy()
            cls_arr = np.asarray(cls).reshape(-1)
            if len(cls_arr):
                class_id = int(cls_arr[0])
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
        # 박스: --plane-mm (기본 3). 마스크: 3mm. 책상 평면에서 이 거리 이내 점은 제거한다.
        if plane is not None:
            cut_mm = MASK_PLANE_MM if use_mask else float(plane_mm)
            roi = roi & plane_foreground_mask(depth, K, plane, height_mm=cut_mm)
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
                quad=quad,
                class_id=class_id,
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
