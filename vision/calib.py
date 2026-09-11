"""Lens / hand-eye JSON I/O. Files live in vision/calib_data/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

VISION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VISION_DIR.parent
CALIB_DIR = VISION_DIR / "calib_data"
INTRINSICS_JSON = CALIB_DIR / "intrinsics.json"
HANDEYE_JSON = CALIB_DIR / "eye_to_hand.json"


def load_K(path: Path | None = None) -> tuple[np.ndarray, Path]:
    source = path or INTRINSICS_JSON
    if not source.is_file():
        raise FileNotFoundError(f"intrinsics.json 없음: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    K = np.array(data["camera_matrix"], dtype=np.float64)
    return K, source


def load_dist(path: Path | None = None) -> tuple[np.ndarray, Path]:
    source = path or INTRINSICS_JSON
    if not source.is_file():
        raise FileNotFoundError(f"intrinsics.json 없음: {source}")
    data = json.loads(source.read_text(encoding="utf-8"))
    dist = np.array(data["dist_coeffs"], dtype=np.float64)
    return dist, source


def load_intrinsics(path: Path | None = None) -> tuple[np.ndarray, np.ndarray, Path]:
    source = path or INTRINSICS_JSON
    K, _ = load_K(source)
    dist, _ = load_dist(source)
    return K, dist, source


def intrinsics_for_rotate180(
    K: np.ndarray,
    dist: np.ndarray | None,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """180° 회전 영상에 맞게 주점·접선왜곡을 뒤집는다. fx, fy, k1, k2는 그대로."""
    k_rot = np.asarray(K, dtype=np.float64).copy()
    k_rot[0, 2] = float(width) - 1.0 - k_rot[0, 2]
    k_rot[1, 2] = float(height) - 1.0 - k_rot[1, 2]
    if dist is None:
        return k_rot, None
    d_rot = np.asarray(dist, dtype=np.float64).reshape(-1).copy()
    if d_rot.size >= 4:
        d_rot[2] *= -1.0
        d_rot[3] *= -1.0
    return k_rot, d_rot


def load_T_base_cam(path: Path | None = None) -> tuple[np.ndarray, Path]:
    """T_base_cam in project base (URDF Rz180°). Re-run hand-eye after frame change."""
    source = path or HANDEYE_JSON
    if not source.is_file():
        raise FileNotFoundError(
            f"eye_to_hand.json 없음: {source}\n"
            "프로젝트 베이스(URDF Rz180°)로 손-눈을 다시 구한 뒤 이 경로에 저장하세요."
        )
    data = json.loads(source.read_text(encoding="utf-8"))
    T = np.array(data["T_base_cam"], dtype=np.float64)
    return T, source


def save_intrinsics(payload: dict, path: Path | None = None) -> Path:
    dest = path or INTRINSICS_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def save_handeye(payload: dict, path: Path | None = None) -> Path:
    dest = path or HANDEYE_JSON
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest
