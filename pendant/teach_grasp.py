#!/usr/bin/env python3
"""물체 배치 + grasp yaml 값을 Meshcat에서 확인·티칭하는 GUI.

권장: 펜던트와 한 프로세스·한 Meshcat.

  python pendant/main.py              # 펜던트만 (사물 위치 칸은 잠김)
  python pendant/main.py --grasp      # 같은 창에서 사물 위치 칸 조작
  python pendant/teach_grasp.py       # 물체 창만 (가상 팔로 P/G 이동)

랜덤 배치 세 자세의 Rx/Ry/Rz (sample_random_place).
각 자세마다 물체 축 기준과 베이스 기준을 둘 다 적는다.
저장·yaml 은 베이스. 「베이스」= extrinsic XYZ, R = Rz @ Ry @ Rx.
「물체 축」= intrinsic XYZ, R = Rx @ Ry @ Rz.

서있기 (세우기). CAD Z 가 베이스 +Z (수직으로 섬).
  물체 축: Rx=0, Ry=0, Rz=−180~180
  베이스:   Rx=0, Ry=0, Rz=−180~180 (물체 축과 같은 값)

눕히기. CAD Y 가 ±베이스 Z 와 나란하다 (납작하게 누움).
  물체 축: Rx=+90 또는 −90, Ry=−180~180, Rz=0
  베이스:   Rx=물체 축 Rx 와 같은 ±90, Ry=0, Rz=(그 Rx 의 부호)×(물체 축 Ry)
  예: 물체 축 (90, 30, 0) → 베이스 (90, 0, 30)
      물체 축 (−90, 30, 0) → 베이스 (−90, 0, −30)
  물체 축 Ry 가 ±180 이면 같은 자세가 접혀, 물체 축이
  (∓90, 0, 180) 쪽으로 다시 읽힐 수 있다. 베이스는 (Rx, 0, ±180).

비스듬히. CAD 축 (−1, 0, −1)/√2 를 베이스 +Z 에 맞춘 뒤(ㅅ),
그 축으로 φ(−180~180) 만큼 돌린다. R = Rz(φ) @ R0, R0 = Rz(180) @ Ry(45) @ Rx(180).
  물체 축: φ마다 Rx·Ry·Rz 가 같이 바뀐다.
           φ=0 → (−180, 45, −180),  φ=90 → (−135, 0, 90)
  베이스:   Rx=180, Ry=45 고정, Rz=wrap(180+φ) (−180~180).
           φ=0 → (180, 45, 180),  φ=90 → (180, 45, −90)
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PENDANT = Path(__file__).resolve().parent
PROJECT = PENDANT.parent
for path in (PROJECT, PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_root_fonts, apply_ui_theme, enable_xft_tk, ui_font  # noqa: E402

enable_xft_tk()
apply_ui_theme()

from yolo.config import (  # noqa: E402
    CAD_YAML,
    REGISTER_REQUEST_JSON,
    REGISTER_STATUS_JSON,
    cad_mesh_path,
    cad_unit,
    quiet_gtk,
)
from yolo.pose.register import _to_mm  # noqa: E402
from motion.pick_place import PickPlaceRunner, build_pick_place_steps  # noqa: E402

# 시뮬에서 물체를 책상에 올려 두는 기본값 (프로젝트 베이스 mm / deg).
# 프로젝트 베이스 = +X 전진 / +Z 위. mesh_rpy Rx180 적용 후 place/grasp.
DEFAULT_PLACE_XYZ_MM = (150.0, 150.0, 2.0)
DEFAULT_PLACE_RPY_DEG = (0.0, 0.0, -90.0)
DEFAULT_GRASP_XYZ_MM = (0.0, 0.0, 0.0)
DEFAULT_GRASP_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_GRIPPER = 50.0
DEFAULT_DROP_XY_MM = (150.0, -100.0)
DEFAULT_APPROACH_D_MM = 10.0
DEFAULT_APPROACH_A_MM = 30.0  # pre → grasp along axis (mm toward CAD origin)
APPROACH_MARKER_RADIUS_M = 0.004  # 4 mm
REG_ACK_S = 2.0
REG_JOB_S = 30.0
REG_POLL_MS = 200
_REG_RUN = "현재 등록 중입니다."
_REG_NO_BOX = "바운딩 박스가 없습니다. 카메라 창을 확인해 주세요."
_REG_FAIL = "등록에 실패했습니다. 카메라 창을 확인해 주세요."
_REG_NO_REPLY = "카메라 창이 응답하지 않습니다. 실행중인지 확인해주세요"
# TCP vs orange pre sphere: turn green when close (debug go-to).
PRE_TCP_MATCH_POS_MM = 1.5
APPROACH_PRE_COLOR = 0xFB923C
APPROACH_PRE_MATCH_COLOR = 0x22C55E
APPROACH_OTHER_COLOR = 0xC2410C
# Random place (project base mm): polar sector ∩ box ∩ r ring ∩ height, mesh above floor.
# x=r·cosθ, y=r·sinθ; θ ∈ (−90°, 90°) → x > 0 (전진) half-plane.
# r_min: INIT 팔과 AABB 겹침 회피. r_max: 먼 코너 IK (r≳271 mm) 회피. 박스는 유지.
RANDOM_XY_ABS_MAX_MM = 200.0
RANDOM_R_MIN_MM = 120.0  # r > 120 mm
RANDOM_R_MAX_MM = 270.0  # r < 270 mm (cuts |x|≈|y|≈200 corners, keeps (200, 0))
RANDOM_R2_MIN_MM2 = RANDOM_R_MIN_MM * RANDOM_R_MIN_MM
RANDOM_R2_MAX_MM2 = RANDOM_R_MAX_MM * RANDOM_R_MAX_MM
RANDOM_THETA_MIN_DEG = -90.0
RANDOM_THETA_MAX_DEG = 90.0
RANDOM_Z_MIN_MM = 0.0
RANDOM_Z_MAX_MM = 40.0
RANDOM_PLACE_MAX_TRIES = 800
# Constrained random-place modes (project base).
STAND_Z_MM = 2.0
STAND_ROLL_DEG = 0.0
STAND_PITCH_DEG = 0.0
# Stand (CAD Z approach): shift P opposite TCP X along object X so P→G clears the lip.
STAND_PRE_SHIFT_NEG_X_MM = 5.0  # TCP X ∥ object +X → P along −X
STAND_PRE_SHIFT_POS_X_MM = 10.0  # TCP X ∥ object −X → P along +X
SLANT_PRE_SHIFT_MM = 15.0  # ㅅ/V: P (and G) opposite TCP X
# Lie (CAD Y approach): extra P at (0.6 lx, same Y as face+d, 0.6 lz), TCP X ∥ CAD (−1,0,−1).
LIE_ALT_XZ_FRAC = 0.4
LIE_ALT_TCP_X_CAD = np.array([-1.0, 0.0, -1.0], dtype=float)
LIE_Z_MM = 17.5
LIE_YAW_DEG = 0.0
LIE_ROLL_CHOICES_DEG = (90.0, -90.0)
SLANT_Z_MM = 34.0
# CAD (−1,0,−1)/√2 ∥ base +Z → ㅅ. (+1,0,+1) would be V.
SLANT_AXIS_CAD = np.array([-1.0, 0.0, -1.0], dtype=float) / np.sqrt(2.0)
RANDOM_PLACE_MODES = ("stand", "lie", "slant", "free")
DISPLAY_MS = 33
RPY_SLIDER_MIN = -180.0
RPY_SLIDER_MAX = 180.0
# Place XYZ sliders (project base mm); match random-place workspace box.
PLACE_XY_SLIDER_MIN = -200.0
PLACE_XY_SLIDER_MAX = 200.0
PLACE_Z_SLIDER_MIN = 0.0
PLACE_Z_SLIDER_MAX = 40.0
RPY_LIVE_APPLY_MS = 40


@dataclass(frozen=True)
class PlacePose:
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class GraspSpec:
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    gripper: float


@dataclass(frozen=True)
class AutoPreResult:
    """Auto pre/grasp in project base. TCP columns: X=roll, Y=pitch, Z=yaw.

    Face axes X/Y/Z: pre at (L+d); diagonal N=(1,0,1)/√2: pre at ±d·N (ㅅ/V).
    Grasp = pre − a along that axis (toward CAD origin).
    """

    T_base: np.ndarray  # pre
    T_grasp_base: np.ndarray  # grasp (a)
    axis_name: str
    point_index: int
    pitch_flipped: bool
    d_mm: float
    a_mm: float
    form: str = ""  # "", "ㅅ", "V" when axis is N; "alt" for lie offset P
    other_pre_base: tuple[float, float, float] | None = None  # unused lie P (m)


# CAD diagonal for ㅅ/V approaches (normalized).
N_CAD = np.array([1.0, 0.0, 1.0], dtype=float) / np.sqrt(2.0)


def _vec3(raw: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback
        return _vec3(parsed, fallback)
    return fallback


def _vec2(raw: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback
        return _vec2(parsed, fallback)
    return fallback


def _scalar(raw: object, fallback: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _parse_model_yaml(path: Path) -> dict:
    """들여쓰기 2칸짜리 작은 yaml. PyYAML 없이 grasp/place 만 읽는다."""
    if not path.is_file():
        return {}
    root: dict = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            section = None
            if value == "":
                section = key
                root[key] = {}
            else:
                root[key] = value.strip("'\"")
            continue
        if section is None or not isinstance(root.get(section), dict):
            continue
        root[section][key] = ast.literal_eval(value) if value else None
    return root


def load_specs(path: Path) -> tuple[PlacePose, GraspSpec]:
    data = _parse_model_yaml(path)
    place_raw = data.get("place") if isinstance(data.get("place"), dict) else {}
    grasp_raw = data.get("grasp") if isinstance(data.get("grasp"), dict) else {}
    place = PlacePose(
        xyz_mm=_vec3(place_raw.get("xyz_mm"), DEFAULT_PLACE_XYZ_MM),
        rpy_deg=_vec3(place_raw.get("rpy_deg"), DEFAULT_PLACE_RPY_DEG),
    )
    grasp = GraspSpec(
        xyz_mm=_vec3(grasp_raw.get("xyz_mm"), DEFAULT_GRASP_XYZ_MM),
        rpy_deg=_vec3(grasp_raw.get("rpy_deg"), DEFAULT_GRASP_RPY_DEG),
        gripper=_scalar(grasp_raw.get("gripper"), DEFAULT_GRIPPER),
    )
    return place, grasp


def load_drop_xy_mm(path: Path) -> tuple[float, float]:
    data = _parse_model_yaml(path)
    drop_raw = data.get("drop") if isinstance(data.get("drop"), dict) else {}
    return _vec2(drop_raw.get("xy_mm"), DEFAULT_DROP_XY_MM)


def pose_to_T(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
    from motion.robot_kinematics import rpy_deg_to_rotmat

    T = np.eye(4)
    T[:3, :3] = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    T[:3, 3] = np.asarray(xyz_mm, dtype=float).reshape(3) / 1000.0
    return T


def T_to_xyzrpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from motion.robot_kinematics import rotmat_to_rpy_deg

    M = np.asarray(T, dtype=float)
    xyz_mm = M[:3, 3] * 1000.0
    rpy_deg = rotmat_to_rpy_deg(M[:3, :3])
    return xyz_mm, rpy_deg


def _wrap_deg(deg: float) -> float:
    """Map degrees into (-180, 180]."""
    return float((float(deg) + 180.0) % 360.0 - 180.0)


def rpy_body_xyz_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """CAD-axis (object) RPY: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
    import pinocchio as pin

    r, p, y = np.deg2rad([float(roll), float(pitch), float(yaw)])
    return np.asarray(
        pin.exp3(np.array([r, 0.0, 0.0]))
        @ pin.exp3(np.array([0.0, p, 0.0]))
        @ pin.exp3(np.array([0.0, 0.0, y])),
        dtype=float,
    )


def rotmat_to_rpy_body_xyz(rot: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rpy_body_xyz_to_rotmat."""
    R = np.asarray(rot, dtype=float).reshape(3, 3)
    sp = float(np.clip(R[0, 2], -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    cp = float(np.cos(pitch))
    if abs(cp) > 1e-8:
        roll = float(np.arctan2(-R[1, 2] / cp, R[2, 2] / cp))
        yaw = float(np.arctan2(-R[0, 1] / cp, R[0, 0] / cp))
    else:
        roll = float(np.arctan2(R[1, 0], R[1, 1]))
        yaw = 0.0
    return (
        _wrap_deg(np.degrees(roll)),
        _wrap_deg(np.degrees(pitch)),
        _wrap_deg(np.degrees(yaw)),
    )


def rpy_extrinsic_to_body_xyz(
    rpy_deg: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """Base extrinsic (pose_to_T) → pendant CAD-axis RPY display."""
    from motion.robot_kinematics import rpy_deg_to_rotmat

    r = np.asarray(rpy_deg, dtype=float).reshape(3)
    return rotmat_to_rpy_body_xyz(rpy_deg_to_rotmat(float(r[0]), float(r[1]), float(r[2])))


def rpy_body_xyz_to_extrinsic(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float]:
    """Pendant CAD-axis RPY → base extrinsic for pose_to_T / yaml."""
    from motion.robot_kinematics import rotmat_to_rpy_deg

    out = rotmat_to_rpy_deg(rpy_body_xyz_to_rotmat(roll, pitch, yaw))
    return float(_wrap_deg(out[0])), float(_wrap_deg(out[1])), float(_wrap_deg(out[2]))


def canonicalize_extrinsic_rpy(
    rpy_deg: tuple[float, float, float] | np.ndarray,
) -> tuple[float, float, float]:
    """같은 회전에 대해 베이스 RPY 숫자를 하나로 고정.

    Ry≈±90 짐벌에서 ``rotmat_to_rpy`` 분기가 갈라져도
    (물체축 RPY → extrinsic) 경로로 다시 풀면 roi_cloud·펜던트 표기가 같아진다.
    """
    return rpy_body_xyz_to_extrinsic(*rpy_extrinsic_to_body_xyz(rpy_deg))


def place_rpy_body_delta(
    rpy_deg: tuple[float, float, float] | np.ndarray,
    *,
    axis: str,
    delta_deg: float,
) -> tuple[float, float, float]:
    """물체(CAD) 원점·축 기준 증분 회전 후 베이스 extrinsic RPY.

    R_new = R @ exp(Δ·e_axis). 평행이동은 건드리지 않음(호출측 xyz 고정).
    axis: ``roll``→CAD X, ``pitch``→CAD Y, ``yaw``→CAD Z.
    """
    import pinocchio as pin
    from motion.robot_kinematics import rpy_deg_to_rotmat, rotmat_to_rpy_deg

    idx = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
    d = _wrap_deg(delta_deg)
    if abs(d) < 1e-12:
        r = np.asarray(rpy_deg, dtype=float).reshape(3)
        return float(r[0]), float(r[1]), float(r[2])
    R = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    w = np.zeros(3, dtype=float)
    w[idx] = np.deg2rad(d)
    R_new = R @ pin.exp3(w)
    out = rotmat_to_rpy_deg(R_new)
    return canonicalize_extrinsic_rpy(
        (_wrap_deg(out[0]), _wrap_deg(out[1]), _wrap_deg(out[2]))
    )


def place_rpy_base_delta(
    rpy_deg: tuple[float, float, float] | np.ndarray,
    *,
    axis: str,
    delta_deg: float,
) -> tuple[float, float, float]:
    """베이스 고정축 증분 회전 후 extrinsic RPY. (물체 원점 유지, R만 변경)

    R_new = exp(Δ·e_axis) @ R. 티칭 슬라이더는 ``place_rpy_body_delta`` 를 쓴다.
    """
    import pinocchio as pin
    from motion.robot_kinematics import rpy_deg_to_rotmat, rotmat_to_rpy_deg

    idx = {"roll": 0, "pitch": 1, "yaw": 2}[axis]
    d = float(delta_deg)
    if abs(d) < 1e-12:
        r = np.asarray(rpy_deg, dtype=float).reshape(3)
        return canonicalize_extrinsic_rpy(r)
    R = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    w = np.zeros(3, dtype=float)
    w[idx] = np.deg2rad(d)
    R_new = pin.exp3(w) @ R
    out = rotmat_to_rpy_deg(R_new)
    return canonicalize_extrinsic_rpy(
        (_wrap_deg(out[0]), _wrap_deg(out[1]), _wrap_deg(out[2]))
    )


def place_axis_alignment_text(
    rpy_deg: tuple[float, float, float] | np.ndarray,
    *,
    tol_deg: float = 15.0,
) -> str:
    """물체 XYZ 축이 프로젝트 베이스 어느 축에 가까운지 한 줄 요약."""
    from motion.robot_kinematics import rpy_deg_to_rotmat

    R = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    names = "XYZ"
    parts: list[str] = []
    for i, n in enumerate(names):
        v = R[:, i]
        j = int(np.argmax(np.abs(v)))
        c = float(np.clip(abs(v[j]), 0.0, 1.0))
        ang = float(np.degrees(np.arccos(c)))
        sign = "+" if v[j] >= 0 else "-"
        mark = "" if ang <= tol_deg else f"~{ang:.0f}°"
        parts.append(f"{n}≈{sign}{names[j]}{mark}")
    return " ".join(parts)


def frames_from_specs(place: PlacePose, grasp: GraspSpec) -> np.ndarray:
    """Return T_base_grasp in project base (+X forward)."""
    T_base_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    T_cad_grasp = pose_to_T(np.array(grasp.xyz_mm), np.array(grasp.rpy_deg))
    return T_base_cad @ T_cad_grasp


def _rot_from_pitch_roll(y_hat: np.ndarray, x_raw: np.ndarray) -> np.ndarray:
    """TCP R with columns [roll=X, pitch=Y, yaw=Z], right-handed."""
    y = np.asarray(y_hat, dtype=float).reshape(3)
    yn = float(np.linalg.norm(y))
    if yn < 1e-12:
        raise ValueError("zero pitch axis")
    y = y / yn
    x = np.asarray(x_raw, dtype=float).reshape(3)
    x = x - y * float(np.dot(x, y))
    xn = float(np.linalg.norm(x))
    if xn < 1e-9:
        # Degenerate: pick any orthonormal.
        tmp = np.array([1.0, 0.0, 0.0]) if abs(y[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = tmp - y * float(np.dot(tmp, y))
        xn = float(np.linalg.norm(x))
    x = x / xn
    z = np.cross(x, y)
    z = z / float(np.linalg.norm(z))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def _pre_ik_ok(kin, q0: np.ndarray, xyz_urdf: np.ndarray, R_urdf: np.ndarray) -> bool:
    return _pre_ik_s6(kin, q0, xyz_urdf, R_urdf) is not None


def _pre_ik_s6(kin, q0: np.ndarray, xyz_urdf: np.ndarray, R_urdf: np.ndarray) -> float | None:
    """Return reached S6 (user deg) if pose is reachable within S6 soft limits."""
    import pinocchio as pin
    from motion.robot_kinematics import JOINT_LIMIT_USER_DEG

    q = np.asarray(q0, dtype=float).copy()
    for _ in range(60):
        q = kin.servo_toward(q, xyz_urdf, R_ref=R_urdf, ori_weight=1.0)
    s6 = float(kin.joints_deg(q)["S6"])
    lo, hi = JOINT_LIMIT_USER_DEG["S6"]
    if s6 < lo - 0.5 or s6 > hi + 0.5:
        return None
    pose = kin.forward_tcp(q)
    err_p = float(np.linalg.norm(pose.xyz_m - np.asarray(xyz_urdf, dtype=float)))
    err_r = float(np.linalg.norm(pin.log3(np.asarray(R_urdf) @ pose.rotation.T)))
    if err_p >= 0.0015 or err_r >= 0.12:
        return None
    return s6


def _tcp_s6(kin, q, p_base: np.ndarray, R_tcp: np.ndarray) -> float | None:
    """Reached S6 (user deg) for TCP pose in project base, or None if unreachable."""
    from motion.base_frame import ee_target_urdf_from_user

    T = np.eye(4)
    T[:3, :3] = np.asarray(R_tcp, dtype=float)
    T[:3, 3] = np.asarray(p_base, dtype=float).reshape(3)
    xyz_mm, rpy = T_to_xyzrpy(T)
    xyz_u, R_u = ee_target_urdf_from_user(xyz_mm, rpy)
    return _pre_ik_s6(kin, q, xyz_u, R_u)


def _pick_tcp_rot(
    *,
    pitch_base: np.ndarray,
    roll_candidates: list[np.ndarray],
    p_base: np.ndarray,
    kin,
    q,
    prefer_min_s6_delta: bool = True,
) -> tuple[np.ndarray, bool]:
    """Build TCP R. Prefer unflipped pitch; among those, min |ΔS6| on roll±.

    Pitch flip (e.g. stand green axis pointing down) is only used if no
    unflipped pitch × roll candidate is IK-reachable within S6 limits.
    """
    from motion.base_frame import ee_target_urdf_from_user

    pitch0 = np.asarray(pitch_base, dtype=float).reshape(3)
    rolls = [np.asarray(r, dtype=float).reshape(3) for r in roll_candidates]
    if kin is None or q is None:
        return _rot_from_pitch_roll(pitch0, rolls[0]), False

    s6_now = float(kin.joints_deg(q)["S6"])

    def _best_for_flip(try_flip: bool) -> tuple[float, np.ndarray, bool] | None:
        pitch_try = pitch0 * (-1.0 if try_flip else 1.0)
        best: tuple[float, np.ndarray, bool] | None = None
        for roll in rolls:
            R_try = _rot_from_pitch_roll(pitch_try, roll)
            T = np.eye(4)
            T[:3, :3] = R_try
            T[:3, 3] = p_base
            xyz_mm, rpy = T_to_xyzrpy(T)
            xyz_u, R_u = ee_target_urdf_from_user(xyz_mm, rpy)
            s6 = _pre_ik_s6(kin, q, xyz_u, R_u)
            if s6 is None:
                continue
            cost = abs(s6 - s6_now) if prefer_min_s6_delta else 0.0
            if best is None or cost < best[0] - 1e-9:
                best = (cost, R_try, try_flip)
        return best

    # Never pick pitch− when pitch+ is reachable (avoids stand pitch-down).
    best = _best_for_flip(False)
    if best is None:
        best = _best_for_flip(True)
    if best is not None:
        return best[1], best[2]
    return _rot_from_pitch_roll(pitch0, rolls[0]), False


def _approach_point_cad_mm(
    cad_u: np.ndarray,
    mins_mm: np.ndarray,
    maxs_mm: np.ndarray,
    d_mm: float,
    *,
    from_origin: bool,
) -> np.ndarray:
    """Pre in CAD mm along unit ``cad_u``.

    Face: AABB support (x·u) + d. Origin-centered span (lx+d) is wrong when
    the CAD origin is near one face (e.g. −X at −2 mm, +X at +40 mm).
    Diagonal N: ``from_origin`` → just d along u.
    """
    u = np.asarray(cad_u, dtype=float).reshape(3)
    nrm = float(np.linalg.norm(u))
    if nrm < 1e-12:
        raise ValueError("zero approach axis")
    u = u / nrm
    d = float(d_mm)
    if from_origin:
        return u * d
    mins = np.asarray(mins_mm, dtype=float).reshape(3)
    maxs = np.asarray(maxs_mm, dtype=float).reshape(3)
    support = 0.0
    for i in range(3):
        support += maxs[i] * u[i] if u[i] > 0.0 else mins[i] * u[i]
    return u * (support + d)


def compute_auto_pre(
    place: PlacePose,
    mins_mm: np.ndarray,
    maxs_mm: np.ndarray,
    d_mm: float,
    a_mm: float = DEFAULT_APPROACH_A_MM,
    *,
    kin=None,
    q=None,
) -> AutoPreResult:
    """Pick approach marker; grasp = pre − a toward CAD origin.

    Axis = most aligned with world +Z (|dot|). Side = that axis pointing
    the same way as world +Z, so pre stays above the object origin.
    """
    T_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    R = T_cad[:3, :3]
    t = T_cad[:3, 3]
    base_z = np.array([0.0, 0.0, 1.0])

    n_base = R @ N_CAD
    axis_dirs = (R[:, 0], R[:, 1], R[:, 2], n_base)
    axis_names = ("X", "Y", "Z", "N")
    dots = [abs(float(np.dot(a, base_z))) for a in axis_dirs]
    axis_i = int(np.argmax(dots))
    axis_name = axis_names[axis_i]

    cad_u = {
        "X": np.array([1.0, 0.0, 0.0]),
        "Y": np.array([0.0, 1.0, 0.0]),
        "Z": np.array([0.0, 0.0, 1.0]),
        "N": N_CAD.copy(),
    }[axis_name]
    if float(np.dot(R @ cad_u, base_z)) < 0.0:
        cad_u = -cad_u
    # Face axes: d past the AABB face in that direction (origin is not centered).
    # N: still ±N·d from CAD origin (ㅅ/V).
    p_cad_mm = _approach_point_cad_mm(
        cad_u, mins_mm, maxs_mm, d_mm, from_origin=(axis_name == "N")
    )
    # 0=+X 1=+Y 2=-Y 3=+Z 4=+N(ㅅ) 5=-N(V)
    if axis_name == "X":
        point_index = 0 if float(cad_u[0]) >= 0.0 else 6
    elif axis_name == "Y":
        point_index = 1 if cad_u[1] > 0.0 else 2
    elif axis_name == "Z":
        point_index = 3 if cad_u[2] > 0.0 else 7
    else:
        point_index = 4 if float(np.dot(cad_u, N_CAD)) > 0.0 else 5

    form = ""
    if axis_name == "N":
        form = "ㅅ" if point_index == 4 else "V"

    nrm = float(np.linalg.norm(p_cad_mm))
    if nrm < 1e-9:
        raise ValueError("pre marker at CAD origin")
    g_cad_mm = p_cad_mm * (1.0 - float(a_mm) / nrm)
    p_base = R @ (p_cad_mm / 1000.0) + t
    g_base = R @ (g_cad_mm / 1000.0) + t

    pitch_base = R @ (p_cad_mm / nrm)
    if axis_name == "N":
        roll_cands = [R[:, 1], -R[:, 1]]
    elif axis_name == "X":
        roll_cands = [R[:, 2], -R[:, 2]]
    elif axis_name == "Y":
        # +N only: −N puts the jaw into the bracket on pre→grasp descent.
        roll_cands = [n_base]
    else:  # Z
        roll_cands = [R[:, 0], -R[:, 0]]

    R_tcp, flip = _pick_tcp_rot(
        pitch_base=pitch_base,
        roll_candidates=roll_cands,
        p_base=p_base,
        kin=kin,
        q=q,
        prefer_min_s6_delta=True,
    )
    # Lateral shift: same Δ on P and G so P→G stays along the approach axis.
    shift = np.zeros(3, dtype=float)
    if axis_name == "Z":
        align = float(np.dot(R_tcp[:, 0], R[:, 0]))
        if abs(align) > 1e-6:
            shift_mm = (
                STAND_PRE_SHIFT_NEG_X_MM if align > 0.0 else STAND_PRE_SHIFT_POS_X_MM
            )
            shift = -np.sign(align) * (shift_mm / 1000.0) * R[:, 0]
    elif axis_name == "N":
        tcp_x = np.asarray(R_tcp[:, 0], dtype=float)
        nrm_x = float(np.linalg.norm(tcp_x))
        if nrm_x > 1e-9:
            shift = -(SLANT_PRE_SHIFT_MM / 1000.0) * (tcp_x / nrm_x)
    p_base = p_base + shift
    g_base = g_base + shift

    other_pre_base = None
    if axis_name == "Y":
        span = np.asarray(maxs_mm, dtype=float).reshape(3) - np.asarray(
            mins_mm, dtype=float
        ).reshape(3)
        p_cad_alt = np.array(
            [
                LIE_ALT_XZ_FRAC * float(span[0]),
                float(p_cad_mm[1]),
                LIE_ALT_XZ_FRAC * float(span[2]),
            ],
            dtype=float,
        )
        g_cad_alt = p_cad_alt - float(a_mm) * cad_u
        p_base_alt = R @ (p_cad_alt / 1000.0) + t
        g_base_alt = R @ (g_cad_alt / 1000.0) + t
        R_alt, flip_alt = _pick_tcp_rot(
            pitch_base=R @ cad_u,
            roll_candidates=[R @ LIE_ALT_TCP_X_CAD],
            p_base=p_base_alt,
            kin=kin,
            q=q,
            prefer_min_s6_delta=True,
        )
        s6_now = float(kin.joints_deg(q)["S6"]) if kin is not None and q is not None else 0.0
        s6_face = (
            _tcp_s6(kin, q, p_base, R_tcp) if kin is not None and q is not None else None
        )
        s6_alt = (
            _tcp_s6(kin, q, p_base_alt, R_alt) if kin is not None and q is not None else None
        )
        pick_alt = False
        if s6_alt is not None and s6_face is None:
            pick_alt = True
        elif s6_alt is not None and s6_face is not None:
            # pitch− is ~180° from INIT and breaks P→G. Prefer pitch+ across face/alt
            # before min |ΔS6| (which used to pick a flipped pre).
            if bool(flip_alt) != bool(flip):
                pick_alt = not bool(flip_alt)
            else:
                pick_alt = abs(s6_alt - s6_now) + 1e-9 < abs(s6_face - s6_now)
        if pick_alt:
            other_pre_base = (float(p_base[0]), float(p_base[1]), float(p_base[2]))
            p_base = p_base_alt
            g_base = g_base_alt
            R_tcp = R_alt
            flip = flip_alt
            form = "alt"
        else:
            other_pre_base = (
                float(p_base_alt[0]),
                float(p_base_alt[1]),
                float(p_base_alt[2]),
            )

    T_base = np.eye(4)
    T_base[:3, :3] = R_tcp
    T_base[:3, 3] = p_base
    T_grasp = np.eye(4)
    T_grasp[:3, :3] = R_tcp
    T_grasp[:3, 3] = g_base
    return AutoPreResult(
        T_base=T_base,
        T_grasp_base=T_grasp,
        axis_name=axis_name,
        point_index=point_index,
        pitch_flipped=flip,
        d_mm=float(d_mm),
        a_mm=float(a_mm),
        form=form,
        other_pre_base=other_pre_base,
    )


def _T_meshcat(T_user: np.ndarray) -> np.ndarray:
    """Project-base pose → Pinocchio/Meshcat (raw URDF) world."""
    from motion.base_frame import T_urdf_from_user

    return T_urdf_from_user(T_user)


def bracket_approach_points_cad_mm(
    mins_mm: np.ndarray,
    maxs_mm: np.ndarray,
    d_mm: float,
) -> np.ndarray:
    """Approach markers in CAD mm from CAD origin.

    Face: d past the AABB face (+X/−X, ±Y, +Z/−Z).
    Diagonal N=(1,0,1)/√2 (ㅅ/V): ±N·d from origin.
    """
    axes = (
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, -1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        N_CAD,
        -N_CAD,
    )
    from_origin = (False, False, False, False, True, True)
    return np.asarray(
        [
            _approach_point_cad_mm(u, mins_mm, maxs_mm, d_mm, from_origin=fo)
            for u, fo in zip(axes, from_origin, strict=True)
        ],
        dtype=float,
    )


def _theta_deg_xy(x: float, y: float) -> float:
    """atan2(y,x) in degrees for sector checks (−90°, 90°) → x > 0.

    Negative atan2 angles (y < 0) stay negative / are lifted above 360 so they
    fail RANDOM_THETA_MIN/MAX, matching x=r·cosθ, y=r·sinθ sampling.
    """
    th = float(np.degrees(np.arctan2(y, x)))
    if th <= RANDOM_THETA_MIN_DEG:
        th += 360.0
    return th


def _theta_in_sector(theta_deg: float) -> bool:
    """True if θ ∈ (RANDOM_THETA_MIN_DEG, RANDOM_THETA_MAX_DEG)."""
    return RANDOM_THETA_MIN_DEG < float(theta_deg) < RANDOM_THETA_MAX_DEG


def _r_max_for_theta_mm(theta_rad: float) -> float:
    """Largest r with |r cosθ|<200, |r sinθ|<200, and r < RANDOM_R_MAX_MM."""
    c = abs(float(np.cos(theta_rad)))
    s = abs(float(np.sin(theta_rad)))
    lim = float(RANDOM_R_MAX_MM) - 1e-3
    if c > 1e-12:
        lim = min(lim, (RANDOM_XY_ABS_MAX_MM - 1e-3) / c)
    if s > 1e-12:
        lim = min(lim, (RANDOM_XY_ABS_MAX_MM - 1e-3) / s)
    return float(lim)


def _place_xy_in_region(x: float, y: float) -> bool:
    if abs(x) >= RANDOM_XY_ABS_MAX_MM or abs(y) >= RANDOM_XY_ABS_MAX_MM:
        return False
    r2 = x * x + y * y
    if r2 <= RANDOM_R2_MIN_MM2 or r2 >= RANDOM_R2_MAX_MM2:
        return False
    return _theta_in_sector(_theta_deg_xy(x, y))


def _place_region_overlay_m(n_arc: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-place XY region as a thin sheet in URDF meters.

    Returns (verts, faces, outline_xyz). Outline is (N, 3) along the outer then
    reversed inner boundary.
    """
    from motion.base_frame import R_URDF_FROM_USER

    n = max(8, int(n_arc))
    thetas = np.linspace(
        np.deg2rad(RANDOM_THETA_MIN_DEG + 1e-3),
        np.deg2rad(RANDOM_THETA_MAX_DEG - 1e-3),
        n,
    )
    r_min = float(RANDOM_R_MIN_MM) / 1000.0
    z = 0.0008
    inner = np.empty((n, 3), dtype=float)
    outer = np.empty((n, 3), dtype=float)
    for i, th in enumerate(thetas):
        r_max = _r_max_for_theta_mm(float(th)) / 1000.0
        c, s = float(np.cos(th)), float(np.sin(th))
        inner[i] = (r_min * c, r_min * s, z)
        outer[i] = (r_max * c, r_max * s, z)
    verts_user = np.vstack([inner, outer])
    faces: list[list[int]] = []
    for i in range(n - 1):
        a, b = i, i + 1
        c_i, d = n + i, n + i + 1
        faces.append([a, d, b])
        faces.append([a, c_i, d])
        faces.append([a, b, d])
        faces.append([a, d, c_i])
    outline_user = np.vstack([outer, inner[::-1]])
    R = R_URDF_FROM_USER
    verts = (R @ verts_user.T).T
    outline = (R @ outline_user.T).T
    return verts, np.asarray(faces, dtype=np.uint32), outline


def _place_origin_in_region(xyz_mm: np.ndarray) -> bool:
    x, y, z = (float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2]))
    if not _place_xy_in_region(x, y):
        return False
    if not (RANDOM_Z_MIN_MM < z < RANDOM_Z_MAX_MM):
        return False
    return True


def _mesh_above_floor(verts_cad_m: np.ndarray, place: PlacePose) -> bool:
    """True if every CAD vertex is at project-base z ≥ −1 mm (desk contact slack)."""
    T = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    pts = np.asarray(verts_cad_m, dtype=float).reshape(-1, 3)
    world = (T[:3, :3] @ pts.T).T + T[:3, 3]
    return bool(np.min(world[:, 2]) >= -0.001)


def _sample_xy_mm(rng: np.random.Generator) -> tuple[float, float] | None:
    """One (x,y) in S1 sector ∩ |x|,|y|<200 ∩ (RANDOM_R_MIN_MM, RANDOM_R_MAX_MM). None if impossible."""
    r_min = float(RANDOM_R_MIN_MM) + 1e-3
    theta_deg = float(rng.uniform(RANDOM_THETA_MIN_DEG + 1e-3, RANDOM_THETA_MAX_DEG - 1e-3))
    theta = np.deg2rad(theta_deg)
    r_max = _r_max_for_theta_mm(theta)
    if r_max <= r_min:
        return None
    r = float(rng.uniform(r_min, r_max))
    x = r * float(np.cos(theta))
    y = r * float(np.sin(theta))
    if not _place_xy_in_region(x, y):
        return None
    return x, y


def _R_map_axis_to_ez(u_cad: np.ndarray) -> np.ndarray:
    """Rotation R with R @ u = +Z (project base)."""
    import pinocchio as pin

    u = np.asarray(u_cad, dtype=float).reshape(3)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise ValueError("zero axis")
    u = u / n
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(pin.Quaternion.FromTwoVectors(u, ez).toRotationMatrix(), dtype=float)


def _rpy_slant_about_s(phi_deg: float) -> tuple[float, float, float]:
    """R = Rz(φ) @ R0, R0 maps SLANT_AXIS_CAD → +Z (ㅅ, free spin about that axis)."""
    from motion.robot_kinematics import rotmat_to_rpy_deg

    R0 = _R_map_axis_to_ez(SLANT_AXIS_CAD)
    phi = np.deg2rad(float(phi_deg))
    c, s = float(np.cos(phi)), float(np.sin(phi))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    rpy = rotmat_to_rpy_deg(Rz @ R0)
    return float(rpy[0]), float(rpy[1]), float(rpy[2])


def sample_random_place(
    verts_cad_m: np.ndarray,
    *,
    mode: str | None = None,
    rng: np.random.Generator | None = None,
    max_tries: int = RANDOM_PLACE_MAX_TRIES,
) -> PlacePose | None:
    """Random place. mode: None/free | stand | lie | slant (floor-safe).

    XY always in S1 sector. Unset/free = full 6D; others lock pose family.
    """
    rng = rng or np.random.default_rng()
    key = (mode or "free").strip().lower()
    if key not in RANDOM_PLACE_MODES:
        key = "free"

    for _ in range(int(max_tries)):
        xy = _sample_xy_mm(rng)
        if xy is None:
            continue
        x, y = xy

        if key == "stand":
            z = STAND_Z_MM
            yaw = float(rng.uniform(-180.0, 180.0))
            rpy = (STAND_ROLL_DEG, STAND_PITCH_DEG, yaw)
        elif key == "lie":
            # CAD-axis body RPY: roll=±90, yaw=0, pitch free → CAD Y ∥ ±base Z
            # (flat ㄴ). Convert to base extrinsic for pose_to_T / yaml.
            z = LIE_Z_MM
            roll = float(rng.choice(LIE_ROLL_CHOICES_DEG))
            pitch = float(rng.uniform(-180.0, 180.0))
            rpy = rpy_body_xyz_to_extrinsic(roll, pitch, LIE_YAW_DEG)
        elif key == "slant":
            z = SLANT_Z_MM
            phi = float(rng.uniform(-180.0, 180.0))
            rpy = _rpy_slant_about_s(phi)
        else:
            z = float(rng.uniform(RANDOM_Z_MIN_MM + 1e-3, RANDOM_Z_MAX_MM - 1e-3))
            rpy = tuple(float(v) for v in rng.uniform(-180.0, 180.0, size=3))

        place = PlacePose(xyz_mm=(x, y, z), rpy_deg=rpy)  # type: ignore[arg-type]
        if key == "free" and not _place_origin_in_region(np.array(place.xyz_mm)):
            continue
        if key != "free" and not _place_xy_in_region(x, y):
            continue
        if not _mesh_above_floor(verts_cad_m, place):
            continue
        return place
    return None


def object_collides_robot(
    kin,
    q: np.ndarray,
    verts_cad_m: np.ndarray,
    place: PlacePose,
) -> bool:
    """AABB of placed CAD vs robot collision geometries (URDF world)."""
    import coal
    import pinocchio as pin
    from motion.base_frame import T_urdf_from_user

    T_user = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    T_urdf = T_urdf_from_user(T_user)
    pts = np.asarray(verts_cad_m, dtype=float).reshape(-1, 3)
    world = (T_urdf[:3, :3] @ pts.T).T + T_urdf[:3, 3]
    mn = world.min(axis=0)
    mx = world.max(axis=0)
    sizes = np.maximum(mx - mn, 1e-4)
    center = 0.5 * (mn + mx)
    box = coal.Box(float(sizes[0]), float(sizes[1]), float(sizes[2]))
    T_box = coal.Transform3s(np.eye(3), center)
    obj = coal.CollisionObject(box, T_box)

    q = np.asarray(q, dtype=float)
    pin.forwardKinematics(kin.model, kin.data, q)
    pin.updateGeometryPlacements(
        kin.model, kin.data, kin.collision_model, kin.collision_data, q
    )
    req = coal.CollisionRequest()
    gd = kin.collision_data
    cm = kin.collision_model
    for i in range(len(cm.geometryObjects)):
        go = cm.geometryObjects[i]
        geom = go.geometry
        if geom is None:
            continue
        oMg = gd.oMg[i]
        rob = coal.CollisionObject(
            geom,
            coal.Transform3s(np.asarray(oMg.rotation), np.asarray(oMg.translation)),
        )
        res = coal.CollisionResult()
        coal.collide(obj, rob, req, res)
        if res.isCollision():
            return True
    return False


def load_cad_mesh_m(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d
    from yolo.pose.register import apply_cad_mesh_frame

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise RuntimeError(f"삼각형 없음: {path}")
    verts_mm = apply_cad_mesh_frame(_to_mm(np.asarray(mesh.vertices), cad_unit(), path), source=path)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    return verts_mm / 1000.0, faces


def _load_user_pick_run():
    """프로젝트 루트의 user_pickandplace.py 를 매번 다시 읽는다."""
    path = PROJECT / "user_pickandplace.py"
    if not path.is_file():
        raise FileNotFoundError(f"파일 없음: {path}")
    spec = importlib.util.spec_from_file_location("user_pickandplace_live", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("user_pickandplace.py 를 읽지 못했습니다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError("user_pickandplace.py 에 run(arm) 함수가 없습니다")
    return run


class TeachGraspGui:
    def __init__(
        self,
        root,
        *,
        visualizer,
        mesh_path: Path,
        vertices_m: np.ndarray,
        faces: np.ndarray,
        place: PlacePose,
        grasp: GraspSpec,
        meshcat_url: str,
        kinematics=None,
        controller=None,
        own_robot: bool = True,
        drive_viz: bool = False,
        embedded: bool = False,
        enabled: bool = True,
        actions_parent=None,
    ) -> None:
        import customtkinter as ctk

        self.root = root
        self._embedded = bool(embedded)
        self._panel_enabled = bool(enabled) if self._embedded else True
        self._panel_labels: list = []
        self._panel_buttons: list = []
        self._mode_cbs: list = []
        self._place_xyz_frame = "base"
        if not self._embedded:
            apply_root_fonts(root)
        self._viz = visualizer
        self._kin = kinematics
        self._ctrl = controller
        self._own_robot = bool(own_robot) and controller is None
        self._drive_viz = bool(drive_viz) and controller is not None
        self._yaml_path = CAD_YAML
        self._grasp_spec = grasp
        self._drop_xy = DEFAULT_DROP_XY_MM
        self._grasp_hide_var = None
        self._region_hide_var = None
        self._cad_loaded = False
        self._auto_pre: AutoPreResult | None = None
        self._pre_tcp_matched: bool | None = None
        self._status: ctk.CTkLabel | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.sliders: dict[str, ctk.CTkSlider] = {}
        self._rpy_sync = False
        self._rpy_apply_after: str | None = None
        self._replay_after: str | None = None
        self._pp_after: str | None = None
        self._reg_after: str | None = None
        self._reg_req_id: str | None = None
        self._reg_ack_deadline = 0.0
        self._reg_job_deadline = 0.0
        self._reg_accepted = False
        self._pp_runner: PickPlaceRunner | None = (
            PickPlaceRunner(controller) if controller is not None else None
        )
        self._user_pp_thread: threading.Thread | None = None
        verts = np.asarray(vertices_m, dtype=float).reshape(-1, 3)
        self._cad_verts_m = verts
        self._cad_mins_mm = verts.min(axis=0) * 1000.0
        self._cad_maxs_mm = verts.max(axis=0) * 1000.0
        self._rng = np.random.default_rng()
        self._collision_warn = False
        self._warn_label: ctk.CTkLabel | None = None
        self._rand_mode: str | None = None
        self._rand_mode_vars: dict[str, object] = {}
        yaml_root = _parse_model_yaml(CAD_YAML)
        approach_d0 = _scalar(yaml_root.get("approach_d_mm"), DEFAULT_APPROACH_D_MM)
        approach_a0 = _scalar(yaml_root.get("approach_a_mm"), DEFAULT_APPROACH_A_MM)
        self._approach_d_mm = float(approach_d0)
        self._approach_a_mm = float(approach_a0)
        drop_xy0 = load_drop_xy_mm(CAD_YAML)
        self._drop_xy = drop_xy0
        # place 저장/적용 = 베이스 extrinsic (_place_rpy_prev, canonicalize).
        # 화면 숫자 = 물체 축 또는 베이스. 버튼으로 표기만 바꾼다.
        self._place_rpy_frame = "body"
        self._place_rpy_prev = canonicalize_extrinsic_rpy(place.rpy_deg)
        shown = self._place_rpy_shown()
        self._place_slider_cmd = dict(shown)

        if not self._embedded:
            root.title("물체 집기 티칭")
            root.minsize(980, 720)
            root.geometry("1080x860")
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            split = ctk.CTkFrame(root, fg_color="transparent")
            split.pack(fill="both", expand=True)
            host = ctk.CTkFrame(split)
            actions_host = ctk.CTkFrame(split)
            host.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
            actions_host.pack(side="left", fill="y", padx=(4, 8), pady=8)
        else:
            host = root
            actions_host = actions_parent if actions_parent is not None else root
        self._host = host
        self._actions_host = actions_host

        # TCP 칸과 같은 시작 높이·같은 줄 간격. 박스는 TCP 행과 같은 CTkFrame.
        title = ctk.CTkLabel(host, text="사물의 위치", anchor="w")
        title.pack(anchor="w", padx=8, pady=(8, 4))
        self._panel_labels.append(title)
        self._place_xyz_seg = self._xyz_jog_frame_bar(host)
        for key, label, val in (
            ("place_x", "x (mm)", place.xyz_mm[0]),
            ("place_y", "y (mm)", place.xyz_mm[1]),
            ("place_z", "z (mm)", place.xyz_mm[2]),
        ):
            self._xyz_row(host, key, float(val), label)

        rpy_head = ctk.CTkFrame(host, fg_color="transparent")
        rpy_head.pack(fill="x", padx=6, pady=(10, 2))
        rpy_title = ctk.CTkLabel(rpy_head, text="Rx/Ry/Rz", width=200, anchor="w")
        rpy_title.pack(side="left", padx=(4, 2))
        self._panel_labels.append(rpy_title)
        self._place_rpy_seg = ctk.CTkSegmentedButton(
            rpy_head,
            values=["base", "OBJ"],
            command=self._on_place_rpy_frame,
            width=148,
        )
        self._place_rpy_seg.set("OBJ")
        self._place_rpy_seg.pack(side="left", padx=(8, 4))
        for key, label, axis in (
            ("place_roll", "Rx", "roll"),
            ("place_pitch", "Ry", "pitch"),
            ("place_yaw", "Rz", "yaw"),
        ):
            self._rpy_row(host, key, float(shown[axis]), label)
        self._rpy_hint = ctk.CTkLabel(
            host,
            text="",
            text_color="#9ca3af",
            anchor="w",
        )
        self._rpy_hint.pack(fill="x", padx=8, pady=(0, 2))
        self._update_rpy_hint()

        hide_row = ctk.CTkFrame(host, fg_color="transparent")
        hide_row.pack(fill="x", padx=8, pady=(2, 2))
        self._grasp_hide_var = ctk.BooleanVar(value=False)
        self._grasp_hide_cb = ctk.CTkCheckBox(
            hide_row,
            text="비활성화 (축 숨김)",
            variable=self._grasp_hide_var,
            command=self._on_grasp_hide_toggle,
            width=160,
        )
        self._grasp_hide_cb.pack(side="left")
        self._region_hide_var = ctk.BooleanVar(value=False)
        self._region_hide_cb = ctk.CTkCheckBox(
            hide_row,
            text="비활성화 (영역 숨김)",
            variable=self._region_hide_var,
            command=self._on_region_hide_toggle,
            width=170,
        )
        self._region_hide_cb.pack(side="left", padx=(12, 0))

        self._panel_labels.append(self._rpy_hint)

        actions = ctk.CTkFrame(actions_host, fg_color="transparent")
        actions.pack(fill="x", padx=14, pady=(4, 8))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        btn_apply = ctk.CTkButton(actions, text="적용", command=self.apply)
        btn_apply.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        btn_reset = ctk.CTkButton(actions, text="물체 초기화", command=self.reset_place)
        btn_reset.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))
        self._panel_buttons.extend((btn_apply, btn_reset))
        self._btn_pre = ctk.CTkButton(actions, text="대기 위치로", command=self.goto_pre)
        self._btn_grasp = ctk.CTkButton(actions, text="집기 위치로", command=self.goto_grasp)
        self._btn_pre.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        self._btn_grasp.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))
        self._btn_seq = ctk.CTkButton(
            actions,
            text="픽앤플레이스",
            command=self.start_pick_place,
            height=56,
            corner_radius=12,
            font=ui_font(18),
        )
        self._btn_seq.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._btn_user_seq = ctk.CTkButton(
            actions,
            text="픽앤플레이스(유저생성)",
            command=self.start_user_pick_place,
            height=56,
            corner_radius=12,
            font=ui_font(18),
        )
        self._btn_user_seq.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._btn_init = ctk.CTkButton(
            actions, text="초기자세로 이동", command=self.snap_init_pose
        )
        self._btn_init.grid(row=4, column=0, columnspan=2, sticky="ew")
        self._panel_buttons.extend(
            (
                self._btn_pre,
                self._btn_grasp,
                self._btn_seq,
                self._btn_user_seq,
                self._btn_init,
            )
        )
        # 픽앤플레이스: motion/pick_place.py 시퀀스.
        if self._ctrl is None:
            self._btn_pre.configure(state="disabled")
            self._btn_grasp.configure(state="disabled")
            self._btn_seq.configure(state="disabled")
            self._btn_user_seq.configure(state="disabled")
            self._btn_init.configure(state="disabled")

        rand_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        rand_row.pack(fill="x", padx=14, pady=(0, 2))
        btn_rand = ctk.CTkButton(
            rand_row,
            text="물체 랜덤 생성",
            width=140,
            command=self.randomize_place,
        )
        btn_rand.pack(side="left")
        self._panel_buttons.append(btn_rand)
        self._warn_label = ctk.CTkLabel(
            rand_row,
            text="",
            anchor="w",
            text_color="#f87171",
        )
        self._warn_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        mode_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(0, 6))
        mode_specs = (
            ("stand", "세우기"),
            ("lie", "눕히기"),
            ("slant", "비스듬히"),
            ("free", "자유 포즈"),
        )
        for key, label in mode_specs:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                mode_row,
                text=label,
                variable=var,
                command=lambda k=key: self._on_rand_mode_toggle(k),
                checkbox_width=18,
                checkbox_height=18,
            )
            cb.pack(side="left", padx=(0, 8))
            self._rand_mode_vars[key] = var
            self._mode_cbs.append(cb)

        foot = ctk.CTkFrame(actions_host, fg_color="transparent")
        foot.pack(fill="x", padx=14, pady=(0, 4))
        self._row(foot, "approach_d", f"{approach_d0:.1f}", "d mm (pre)")
        self._row(foot, "approach_a", f"{approach_a0:.1f}", "a mm (grasp)")
        self.entries["approach_d"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_d"].bind("<Return>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<Return>", lambda _e: self._schedule_live_apply())

        self._status = ctk.CTkLabel(actions_host, text="", anchor="w")
        self._status.pack(fill="x", padx=14, pady=(0, 6))
        btn_load = ctk.CTkButton(
            actions_host,
            text="물체 위치 불러오기",
            command=self.load_registered_place,
            fg_color="#86EFAC",
            hover_color="#4ADE80",
            text_color="#14532D",
        )
        btn_load.pack(fill="x", padx=14, pady=(0, 14))
        self._panel_buttons.append(btn_load)

        if self._embedded and not self._panel_enabled:
            self._lock_place_panel()
            return

        if not self._embedded:
            root.bind("<Return>", lambda _e: self.apply())
        self._draw_static(vertices_m, faces)
        self.apply()
        # Color pre sphere while TCP moves; pendant owns display(q) when not drive_viz.
        if self._ctrl is not None:
            self._schedule_display()

    def _xyz_jog_frame_bar(self, parent):
        """TCP 패널의 XYZ jog frame 과 같은 자리. 숫자는 그대로 둔다."""
        import customtkinter as ctk

        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(0, 2))
        lab = ctk.CTkLabel(bar, text="XYZ jog frame", width=200, anchor="w")
        lab.pack(side="left", padx=(4, 2))
        self._panel_labels.append(lab)
        seg = ctk.CTkSegmentedButton(
            bar,
            values=["base", "TCP"],
            command=self._on_place_xyz_frame,
            width=148,
        )
        seg.set("base")
        seg.pack(side="left", padx=(8, 4))
        return seg

    def _on_place_xyz_frame(self, value: str) -> None:
        if self._embedded and not self._panel_enabled:
            return
        self._place_xyz_frame = "tcp" if str(value) == "TCP" else "base"

    def _axis_row(self, parent):
        """TCP `_cart_row` 와 같은 프레임. 높이는 조그 버튼(32)에 맞춘다."""
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=3, padx=6)
        return row

    def _section(self, parent, title: str) -> None:
        import customtkinter as ctk

        lab = ctk.CTkLabel(parent, text=title, anchor="w", font=ui_font(15, "bold"))
        lab.pack(fill="x", padx=8, pady=(8, 2))
        self._panel_labels.append(lab)

    def _grasp_section_hidden(self) -> bool:
        var = getattr(self, "_grasp_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_grasp_hide_toggle(self) -> None:
        self.apply()

    def _region_hidden(self) -> bool:
        var = getattr(self, "_region_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_region_hide_toggle(self) -> None:
        if self._region_hidden():
            self._hide_region_overlays()
        else:
            self._show_region_overlays()

    def _hide_grasp_overlays(self) -> None:
        for name in ("grasp", "grasp_marker", "approach"):
            self._viz.clear_overlay(name)

    def _hide_region_overlays(self) -> None:
        for name in ("place_region", "place_region_edge"):
            self._viz.clear_overlay(name)

    def _show_region_overlays(self) -> None:
        region_v, region_f, region_loop = _place_region_overlay_m()
        self._viz.set_overlay_mesh(
            "place_region",
            region_v,
            region_f,
            color=0x86EFAC,
            opacity=0.28,
        )
        self._viz.set_overlay_lines(
            "place_region_edge",
            region_loop,
            color=0x4ADE80,
            closed=True,
        )

    def _xyzrpy_block(
        self,
        parent,
        prefix: str,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
    ) -> None:
        for key, label, val in (
            (f"{prefix}_x", "x mm", xyz[0]),
            (f"{prefix}_y", "y mm", xyz[1]),
            (f"{prefix}_z", "z mm", xyz[2]),
        ):
            if prefix == "place":
                self._xyz_row(parent, key, float(val), label)
            else:
                self._row(parent, key, f"{val:.1f}", label)
        for key, label, val in (
            (f"{prefix}_roll", "Rx deg", rpy[0]),
            (f"{prefix}_pitch", "Ry deg", rpy[1]),
            (f"{prefix}_yaw", "Rz deg", rpy[2]),
        ):
            self._rpy_row(parent, key, float(val), label)

    def _row(self, parent, key: str, value: str, label: str) -> None:
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=140)
        ent.insert(0, value)
        ent.pack(side="left", padx=6)
        self.entries[key] = ent

    def _slider_limits(self, key: str) -> tuple[float, float]:
        if key in ("place_x", "place_y"):
            return PLACE_XY_SLIDER_MIN, PLACE_XY_SLIDER_MAX
        if key == "place_z":
            return PLACE_Z_SLIDER_MIN, PLACE_Z_SLIDER_MAX
        return RPY_SLIDER_MIN, RPY_SLIDER_MAX

    def _xyz_row(self, parent, key: str, value: float, label: str) -> None:
        import customtkinter as ctk

        lo, hi = self._slider_limits(key)
        row = self._axis_row(parent)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=72, height=32)
        ent.insert(0, f"{value:.1f}")
        ent.pack(side="left", padx=(6, 4))
        steps = max(int(round(hi - lo)), 1)
        slider = ctk.CTkSlider(
            row,
            from_=lo,
            to=hi,
            number_of_steps=steps,
            command=lambda v, k=key: self._on_xyz_slider(k, v),
        )
        self._rpy_sync = True
        try:
            slider.set(float(np.clip(value, lo, hi)))
        finally:
            self._rpy_sync = False
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entries[key] = ent
        self.sliders[key] = slider
        ent.bind("<FocusOut>", lambda _e, k=key: self._on_xyz_entry(k))
        ent.bind("<Return>", lambda _e, k=key: self._on_xyz_entry(k))

    def _rpy_row(self, parent, key: str, value: float, label: str) -> None:
        import customtkinter as ctk

        row = self._axis_row(parent)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=72, height=32)
        ent.insert(0, f"{value:.1f}")
        ent.pack(side="left", padx=(6, 4))
        slider = ctk.CTkSlider(
            row,
            from_=RPY_SLIDER_MIN,
            to=RPY_SLIDER_MAX,
            number_of_steps=360,
            command=lambda v, k=key: self._on_rpy_slider(k, v),
        )
        self._rpy_sync = True
        try:
            slider.set(float(np.clip(value, RPY_SLIDER_MIN, RPY_SLIDER_MAX)))
        finally:
            self._rpy_sync = False
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entries[key] = ent
        self.sliders[key] = slider
        ent.bind("<FocusOut>", lambda _e, k=key: self._on_rpy_entry(k))
        ent.bind("<Return>", lambda _e, k=key: self._on_rpy_entry(k))

    def _on_xyz_slider(self, key: str, value: float) -> None:
        if self._rpy_sync:
            return
        self._rpy_sync = True
        try:
            ent = self.entries[key]
            ent.delete(0, "end")
            ent.insert(0, f"{float(value):.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _on_xyz_entry(self, key: str) -> None:
        if self._rpy_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        lo, hi = self._slider_limits(key)
        val = float(np.clip(val, lo, hi))
        self._rpy_sync = True
        try:
            self.sliders[key].set(val)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _place_rpy_shown(self) -> dict[str, float]:
        """현재 버튼 표기로 푼 Rx/Ry/Rz."""
        if self._place_rpy_frame == "base":
            rpy = self._place_rpy_prev
            return {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])}
        roll, pitch, yaw = rpy_extrinsic_to_body_xyz(self._place_rpy_prev)
        return {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)}

    def _update_rpy_hint(self) -> None:
        if self._place_rpy_frame == "base":
            text = "슬라이더 = 베이스 고정축 회전. 숫자는 저장되는 베이스 각."
        else:
            text = "슬라이더 = 물체에 붙은 축 회전. Rz만 돌리면 Rz 숫자만 바뀌는 건 정상."
        self._rpy_hint.configure(text=text)

    def _on_place_rpy_frame(self, value: str) -> None:
        if self._embedded and not self._panel_enabled:
            return
        frame = "base" if str(value) == "base" else "body"
        if frame == self._place_rpy_frame:
            return
        self._place_rpy_frame = frame
        self._sync_place_rpy_ui()
        self._update_rpy_hint()

    def _sync_place_rpy_ui(self, *, keep_axis: str | None = None) -> None:
        """보이는 칸을 현재 표기로 맞춘다. 드래그 중인 축 슬라이더는 유지."""
        extracted = self._place_rpy_shown()
        self._rpy_sync = True
        try:
            for axis, val in extracted.items():
                key = f"place_{axis}"
                if key in self.entries:
                    self.entries[key].delete(0, "end")
                    self.entries[key].insert(0, f"{val:.1f}")
                if axis == keep_axis:
                    continue
                self._place_slider_cmd[axis] = val
                if key in self.sliders:
                    lo, hi = self._slider_limits(key)
                    self.sliders[key].set(float(np.clip(val, lo, hi)))
        finally:
            self._rpy_sync = False

    def _nudge_place_rpy(
        self, axis: str, new_axis_val: float, *, absolute: bool = False
    ) -> None:
        """Rx/Ry/Rz 조작. xyz(물체 원점)는 건드리지 않는다. 저장은 베이스 각."""
        axis = {"roll": "roll", "pitch": "pitch", "yaw": "yaw"}[axis]
        new_axis_val = float(new_axis_val)
        if absolute:
            cmd = dict(self._place_slider_cmd)
            cmd[axis] = new_axis_val
            if self._place_rpy_frame == "base":
                self._place_rpy_prev = canonicalize_extrinsic_rpy(
                    (float(cmd["roll"]), float(cmd["pitch"]), float(cmd["yaw"]))
                )
            else:
                self._place_rpy_prev = rpy_body_xyz_to_extrinsic(
                    float(cmd["roll"]), float(cmd["pitch"]), float(cmd["yaw"])
                )
            self._place_slider_cmd = cmd
            self._sync_place_rpy_ui()
        else:
            old = float(self._place_slider_cmd[axis])
            delta = new_axis_val - old
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0
            if self._place_rpy_frame == "base":
                self._place_rpy_prev = place_rpy_base_delta(
                    self._place_rpy_prev, axis=axis, delta_deg=delta
                )
            else:
                self._place_rpy_prev = place_rpy_body_delta(
                    self._place_rpy_prev, axis=axis, delta_deg=delta
                )
            self._place_slider_cmd[axis] = new_axis_val
            self._sync_place_rpy_ui(keep_axis=axis)
        self._schedule_live_apply()

    def _on_rpy_slider(self, key: str, value: float) -> None:
        if self._rpy_sync:
            return
        if key.startswith("place_"):
            self._nudge_place_rpy(
                key.removeprefix("place_"), float(value), absolute=False
            )
            return
        self._rpy_sync = True
        try:
            ent = self.entries[key]
            ent.delete(0, "end")
            ent.insert(0, f"{float(value):.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _on_rpy_entry(self, key: str) -> None:
        if self._rpy_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        val = float(np.clip(val, RPY_SLIDER_MIN, RPY_SLIDER_MAX))
        if key.startswith("place_"):
            self._nudge_place_rpy(
                key.removeprefix("place_"), val, absolute=True
            )
            return
        self._rpy_sync = True
        try:
            self.sliders[key].set(val)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _schedule_live_apply(self) -> None:
        if self._rpy_apply_after is not None:
            try:
                self.root.after_cancel(self._rpy_apply_after)
            except Exception:
                pass
        self._rpy_apply_after = self.root.after(RPY_LIVE_APPLY_MS, self._live_apply)

    def _live_apply(self) -> None:
        self._rpy_apply_after = None
        try:
            self.apply()
        except Exception:
            pass

    def _set_status(self, text: str, *, error: bool = False, info: bool = False) -> None:
        if self._status is None:
            return
        if error:
            color = "#f87171"
        elif info:
            color = "#e5e7eb"
        else:
            color = "#86efac"
        self._status.configure(text=text, text_color=color)

    def _read_float(self, key: str) -> float:
        return float(self.entries[key].get().strip())

    def _read_place_grasp(self) -> tuple[PlacePose, GraspSpec]:
        place = PlacePose(
            xyz_mm=(
                self._read_float("place_x"),
                self._read_float("place_y"),
                self._read_float("place_z"),
            ),
            rpy_deg=tuple(float(x) for x in self._place_rpy_prev),
        )
        return place, self._grasp_spec

    def _fill(self, place: PlacePose, grasp: GraspSpec) -> None:
        self._fill_place(place)
        self._fill_grasp(grasp)

    def _set_entry(self, key: str, val: float) -> None:
        self.entries[key].delete(0, "end")
        self.entries[key].insert(0, f"{val:.1f}")
        if key in self.sliders:
            lo, hi = self._slider_limits(key)
            clipped = float(np.clip(val, lo, hi))
            self._rpy_sync = True
            try:
                self.sliders[key].set(clipped)
            finally:
                self._rpy_sync = False

    def _fill_place(self, place: PlacePose) -> None:
        # 내부=베이스 extrinsic. 칸 숫자는 현재 물체 축/베이스 표기.
        self._place_rpy_prev = canonicalize_extrinsic_rpy(place.rpy_deg)
        shown = self._place_rpy_shown()
        self._place_slider_cmd = dict(shown)
        mapping = {
            "place_x": place.xyz_mm[0],
            "place_y": place.xyz_mm[1],
            "place_z": place.xyz_mm[2],
            "place_roll": shown["roll"],
            "place_pitch": shown["pitch"],
            "place_yaw": shown["yaw"],
        }
        for key, val in mapping.items():
            self._set_entry(key, val)

    def _fill_grasp(self, grasp: GraspSpec) -> None:
        self._grasp_spec = grasp

    def _draw_static(self, vertices_m: np.ndarray, faces: np.ndarray) -> None:
        table_t = 0.004
        T_table = np.eye(4)
        T_table[2, 3] = -table_t / 2.0
        self._viz.set_overlay_box(
            "table",
            (0.80, 0.80, table_t),
            T_table,
            color=0x2F2F35,
            opacity=0.45,
        )
        if not self._region_hidden():
            self._show_region_overlays()
        self._viz.set_overlay_mesh("cad", vertices_m, faces, color=0xC4C4C8, opacity=0.92)
        self._cad_loaded = True

    def _show_robot(self, gripper_100: float) -> None:
        if self._kin is None:
            return
        from motion.hw_controller import grip_100_to_user
        from motion.robot_kinematics import GRIPPER_JOINT, HOME_JOINTS_DEG, URDF_JOINT_NAMES

        joints = {name: float(HOME_JOINTS_DEG[name]) for name in URDF_JOINT_NAMES}
        joints[GRIPPER_JOINT] = grip_100_to_user(gripper_100)
        self._viz.display(self._kin.q_from_deg(joints))

    def _lock_place_panel(self) -> None:
        gray = "#6b7280"
        for lab in self._panel_labels:
            try:
                lab.configure(text_color=gray)
            except Exception:
                pass
        for ent in self.entries.values():
            ent.configure(text_color=gray, state="disabled")
        for slider in self.sliders.values():
            slider.configure(state="disabled")
        self._place_rpy_seg.configure(state="disabled")
        self._place_xyz_seg.configure(state="disabled")
        for cb in (self._grasp_hide_cb, self._region_hide_cb, *self._mode_cbs):
            cb.configure(state="disabled", text_color_disabled=gray)
        for btn in self._panel_buttons:
            btn.configure(state="disabled")

    def _schedule_display(self) -> None:
        if self._ctrl is None:
            return
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        st = self._ctrl.snapshot()
        if self._drive_viz:
            self._viz.display(st.q)
        self._update_pre_match_color(st)
        self.root.after(DISPLAY_MS, self._schedule_display)

    def _tcp_matches_pre(self, st) -> bool:
        """True if TCP is on the orange pre sphere (project-base position)."""
        auto = self._auto_pre
        if auto is None:
            return False
        err_p = float(
            np.linalg.norm(np.asarray(st.pose.xyz_m, dtype=float) - auto.T_base[:3, 3])
        )
        return err_p * 1000.0 < PRE_TCP_MATCH_POS_MM

    def _draw_approach_markers(self, *, matched: bool) -> None:
        auto = self._auto_pre
        if auto is None:
            return
        T_p = _T_meshcat(auto.T_base)
        centers = [T_p[:3, 3]]
        colors = [APPROACH_PRE_MATCH_COLOR if matched else APPROACH_PRE_COLOR]
        if auto.other_pre_base is not None:
            T_other = np.eye(4)
            T_other[:3, 3] = np.asarray(auto.other_pre_base, dtype=float)
            centers.append(_T_meshcat(T_other)[:3, 3])
            colors.append(APPROACH_OTHER_COLOR)
        self._viz.clear_overlay("approach_markers")
        self._viz.set_overlay_spheres(
            "approach_markers",
            np.stack(centers, axis=0),
            radius_m=APPROACH_MARKER_RADIUS_M,
            colors=colors,
        )

    def _update_pre_match_color(self, st) -> None:
        if self._auto_pre is None:
            return
        matched = self._tcp_matches_pre(st)
        if matched is self._pre_tcp_matched:
            return
        self._pre_tcp_matched = matched
        self._draw_approach_markers(matched=matched)

    def _read_da(self) -> tuple[float, float]:
        try:
            d_mm = self._read_float("approach_d")
        except (KeyError, ValueError):
            d_mm = self._approach_d_mm
        try:
            a_mm = self._read_float("approach_a")
        except (KeyError, ValueError):
            a_mm = self._approach_a_mm
        return float(d_mm), float(a_mm)

    def _compute_auto(self, place: PlacePose) -> AutoPreResult:
        d_mm, a_mm = self._read_da()
        kin = self._kin
        if kin is None and self._ctrl is not None:
            kin = getattr(self._ctrl, "_kin", None)
        q = None
        if self._ctrl is not None:
            q = self._ctrl.snapshot().q
        elif kin is not None:
            q = kin.q_home()
        return compute_auto_pre(
            place,
            self._cad_mins_mm,
            self._cad_maxs_mm,
            d_mm,
            a_mm,
            kin=kin,
            q=q,
        )

    def _cancel_register_poll(self) -> None:
        if self._reg_after is None:
            return
        try:
            self.root.after_cancel(self._reg_after)
        except Exception:
            pass
        self._reg_after = None

    def load_registered_place(self) -> None:
        """카메라 창에 c 와 같은 전체 등록을 요청하고, 끝난 자세만 칸에 넣는다."""
        if self._embedded and not self._panel_enabled:
            return
        self._cancel_register_poll()
        req_id = uuid.uuid4().hex
        self._reg_req_id = req_id
        now = time.monotonic()
        self._reg_ack_deadline = now + REG_ACK_S
        self._reg_job_deadline = now + REG_JOB_S
        self._reg_accepted = False
        try:
            REGISTER_REQUEST_JSON.parent.mkdir(parents=True, exist_ok=True)
            tmp = REGISTER_REQUEST_JSON.with_suffix(".json.tmp")
            tmp.write_text(json.dumps({"id": req_id}), encoding="utf-8")
            tmp.replace(REGISTER_REQUEST_JSON)
        except OSError:
            self._set_status(_REG_NO_REPLY, error=True)
            return
        self._set_status(_REG_RUN, info=True)
        self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def _poll_register(self) -> None:
        self._reg_after = None
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        req_id = self._reg_req_id
        if not req_id:
            return
        now = time.monotonic()
        data = None
        try:
            data = json.loads(REGISTER_STATUS_JSON.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            data = None
        if isinstance(data, dict) and data.get("id") == req_id:
            state = str(data.get("state") or "")
            if state == "run":
                self._reg_accepted = True
                self._set_status(_REG_RUN, info=True)
            elif state == "busy":
                self._set_status(_REG_RUN, info=True)
                return
            elif state == "no_box":
                self._set_status(_REG_NO_BOX, error=True)
                return
            elif state == "fail":
                self._set_status(_REG_FAIL, error=True)
                return
            elif state == "ok":
                place = self._place_from_register(data)
                if place is None:
                    self._set_status(_REG_FAIL, error=True)
                    return
                self._fill_place(place)
                self.apply()
                self._set_status(
                    f"불러옴  xyz=({place.xyz_mm[0]:.1f}, {place.xyz_mm[1]:.1f}, {place.xyz_mm[2]:.1f})"
                )
                return
        if not self._reg_accepted and now >= self._reg_ack_deadline:
            self._set_status(_REG_NO_REPLY, error=True)
            return
        if self._reg_accepted and now >= self._reg_job_deadline:
            self._set_status(_REG_NO_REPLY, error=True)
            return
        self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def _place_from_register(self, data: dict) -> PlacePose | None:
        xyz = data.get("xyz_mm")
        rpy = data.get("rpy_deg")
        if not isinstance(xyz, list) or not isinstance(rpy, list):
            return None
        if len(xyz) < 3 or len(rpy) < 3:
            return None
        try:
            return PlacePose(
                xyz_mm=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
                rpy_deg=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
            )
        except (TypeError, ValueError):
            return None

    def apply(self) -> None:
        if self._embedded and not self._panel_enabled:
            return
        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        d_mm, a_mm = self._read_da()
        auto = self._compute_auto(place)
        self._auto_pre = auto
        T_base_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
        T_cad = _T_meshcat(T_base_cad)
        T_p = _T_meshcat(auto.T_base)
        T_g = _T_meshcat(auto.T_grasp_base)

        if self._cad_loaded:
            self._viz.set_overlay_transform("cad", T_cad)
        self._viz.set_overlay_axes("cad_axes", T_cad, scale=0.02, labels=False)
        self._viz.clear_overlay("base_at_obj")
        self._viz.set_overlay_axes("pregrasp", T_p, scale=0.02)
        # Virtual N axis (x=z, y=0) only while that approach is selected.
        if auto.axis_name == "N":
            n_world = T_cad[:3, :3] @ N_CAD
            o = T_cad[:3, 3]
            half = max(float(d_mm), 20.0) / 1000.0
            self._viz.set_overlay_segment(
                "n_axis",
                o - n_world * half,
                o + n_world * half,
                color=0xC026D3,
            )
        else:
            self._viz.clear_overlay("n_axis")
        matched = False
        if self._ctrl is not None:
            matched = self._tcp_matches_pre(self._ctrl.snapshot())
        self._pre_tcp_matched = matched
        self._draw_approach_markers(matched=matched)
        if self._grasp_section_hidden():
            self._hide_grasp_overlays()
        else:
            self._viz.set_overlay_axes("grasp", T_g, scale=0.0225)
            self._viz.set_overlay_segment("approach", T_p[:3, 3], T_g[:3, 3])
            self._viz.set_overlay_spheres(
                "grasp_marker",
                T_g[:3, 3].reshape(1, 3),
                radius_m=APPROACH_MARKER_RADIUS_M,
                color=0x22C55E,
            )
        if self._own_robot:
            self._show_robot(grasp.gripper)

    def _set_collision_warn(self, text: str) -> None:
        self._collision_warn = bool(text)
        if self._warn_label is not None:
            self._warn_label.configure(text=text)

    def _on_rand_mode_toggle(self, key: str) -> None:
        """One mode at a time; others stay clickable (no lock). Click again to clear."""
        var = self._rand_mode_vars[key]
        selected = bool(var.get())
        if selected:
            self._rand_mode = key
            for k, other in self._rand_mode_vars.items():
                if k != key:
                    other.set(False)
        else:
            self._rand_mode = None

    def _active_rand_mode(self) -> str:
        """Unset checkbox → free (same as 자유 포즈)."""
        m = self._rand_mode
        return m if m in RANDOM_PLACE_MODES else "free"

    def randomize_place(self) -> None:
        """Random place by selected mode (default/unset = free). Floor-safe."""
        self._set_collision_warn("")
        mode = self._active_rand_mode()
        place = sample_random_place(self._cad_verts_m, mode=mode, rng=self._rng)
        if place is None:
            self._set_status(
                f"랜덤 실패({mode}): {RANDOM_PLACE_MAX_TRIES}회 내 바닥/영역 만족 샘플 없음",
                error=True,
            )
            return
        self._fill_place(place)
        self.apply()

        kin = self._kin
        if kin is None and self._ctrl is not None:
            kin = getattr(self._ctrl, "_kin", None)
        q = None
        if self._ctrl is not None:
            q = self._ctrl.snapshot().q
        elif kin is not None:
            q = kin.q_home()

        collided = False
        if kin is not None and q is not None:
            try:
                collided = object_collides_robot(kin, q, self._cad_verts_m, place)
            except Exception as exc:
                self._set_status(
                    f"랜덤({mode}) xyz={list(np.round(place.xyz_mm, 1))}  "
                    f"rpy={list(np.round(place.rpy_deg, 1))}  "
                    f"(충돌검사 실패: {exc})",
                    error=True,
                )
                return

        mode_ko = {
            "stand": "세우기",
            "lie": "눕히기",
            "slant": "비스듬히",
            "free": "자유",
        }.get(mode, mode)
        if collided:
            self._set_collision_warn("경고: 물체가 현재 로봇과 충돌합니다. 다시 랜덤 생성하세요.")
            self._set_status(
                f"랜덤({mode_ko}/충돌) xyz={list(np.round(place.xyz_mm, 1))}  "
                f"rpy={list(np.round(place.rpy_deg, 1))}",
                error=True,
            )
        else:
            self._set_status(
                f"랜덤({mode_ko}) xyz={list(np.round(place.xyz_mm, 1))}  "
                f"rpy={list(np.round(place.rpy_deg, 1))}"
            )

    def reset_place(self) -> None:
        place, _ = load_specs(self._yaml_path)
        self._fill_place(place)
        self.apply()
        self._set_status(f"물체 초기화: {self._yaml_path.name} place")

    def reset_grasp(self) -> None:
        _, grasp = load_specs(self._yaml_path)
        self._fill_grasp(grasp)
        self.apply()
        self._set_status(f"grasp 초기화: {self._yaml_path.name} grasp")

    def _goto_T(self, T: np.ndarray, *, label: str) -> float:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return 0.0
        self.apply()
        xyz_mm, rpy_deg = T_to_xyzrpy(T)
        duration_s = self._ctrl.start_ee_goto(xyz_mm, rpy_deg)
        self._set_status(
            f"{label} 이동  xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
            f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]  "
            f"({duration_s:.1f}s)"
        )
        return float(duration_s)

    def goto_pre(self) -> None:
        try:
            place, _grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        flip_txt = "pitch−" if auto.pitch_flipped else "pitch+"
        form_txt = f" {auto.form}" if auto.form else ""
        self._goto_T(auto.T_base, label=f"대기 위치 ({auto.axis_name}{form_txt} {flip_txt})")

    def goto_grasp(self) -> None:
        try:
            place, _grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        flip_txt = "pitch−" if auto.pitch_flipped else "pitch+"
        form_txt = f" {auto.form}" if auto.form else ""
        self._goto_T(
            auto.T_grasp_base,
            label=f"집기 위치 ({auto.axis_name}{form_txt} a={auto.a_mm:.0f} {flip_txt})",
        )

    def replay_pg(self) -> None:
        self.start_pick_place()

    def start_pick_place(self) -> None:
        if self._ctrl is None or self._pp_runner is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 실행 중", error=True)
            return
        if self._pp_runner.busy():
            self._set_status("픽앤플레이스 이미 실행 중", error=True)
            return
        try:
            place, _grasp = self._read_place_grasp()
            drop_xy = self._drop_xy
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
        g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
        steps = build_pick_place_steps(
            p_xyz_mm=p_xyz,
            p_rpy_deg=p_rpy,
            g_xyz_mm=g_xyz,
            g_rpy_deg=g_rpy,
            drop_xy_mm=np.array(drop_xy, dtype=float),
        )
        if not self._pp_runner.start(steps):
            self._set_status(self._pp_runner.status, error=True)
            return
        self._set_status(f"픽앤플레이스  {self._pp_runner.step_name}")
        self._tick_pick_place()

    def _tick_pick_place(self) -> None:
        self._pp_after = None
        runner = self._pp_runner
        if runner is None:
            return
        phase = runner.tick()
        if phase == "running":
            self._set_status(f"픽앤플레이스  {runner.step_name}")
            self._pp_after = self.root.after(50, self._tick_pick_place)
            return
        if phase == "done":
            self._set_status("픽앤플레이스 완료")
            return
        if phase == "fault":
            self._set_status(f"픽앤플레이스 실패: {runner.status}", error=True)

    def _user_pick_running(self) -> bool:
        thread = self._user_pp_thread
        return thread is not None and thread.is_alive()

    def start_user_pick_place(self) -> None:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 이미 실행 중", error=True)
            return
        if self._pp_runner is not None and self._pp_runner.busy():
            self._set_status("픽앤플레이스 실행 중", error=True)
            return
        self._user_pp_thread = threading.Thread(
            target=self._user_pick_worker, name="user-pick-place", daemon=True
        )
        self._user_pp_thread.start()
        self._set_status("픽앤플레이스(유저생성) 실행")

    def abort_user_pick_place(self) -> None:
        if not self._user_pick_running() or self._ctrl is None:
            return
        self._ctrl.stop_all()

    def _user_pick_worker(self) -> None:
        try:
            run = _load_user_pick_run()
            from motion.arm import Arm

            run(Arm.attach(self._ctrl))
        except Exception as exc:
            self._post_status(f"픽앤플레이스(유저생성) 실패: {exc}", error=True)
        else:
            self._post_status("픽앤플레이스(유저생성) 완료")

    def _post_status(self, text: str, *, error: bool = False) -> None:
        def apply() -> None:
            try:
                self._set_status(text, error=error)
            except Exception:
                pass

        try:
            self.root.after(0, apply)
        except Exception:
            pass

    def snap_init_pose(self) -> None:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        how = self._ctrl.snap_init_pose()
        if how == "snap":
            self._set_status("초기자세로 즉시 이동 (가상)")
        elif how == "goto":
            self._set_status("초기자세로 이동 중 (실기 · 관절 go-to)")
        else:
            self._set_status("초기자세 이동 실패 (E-stop/충돌 등)", error=True)

    def on_close(self) -> None:
        self.abort_user_pick_place()
        if self._pp_runner is not None and self._pp_runner.busy():
            self._pp_runner.abort("창 닫힘")
        if self._pp_after is not None:
            try:
                self.root.after_cancel(self._pp_after)
            except Exception:
                pass
            self._pp_after = None
        if self._replay_after is not None:
            try:
                self.root.after_cancel(self._replay_after)
            except Exception:
                pass
            self._replay_after = None
        self._cancel_register_poll()
        self.root.destroy()


def attach_teach_panel(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    enabled: bool = False,
    actions_parent=None,
) -> TeachGraspGui | None:
    """로봇 펜던트 열에 사물 위치 블록을 붙인다. enabled 가 아니면 표시만 하고 잠근다."""
    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_cad_mesh_m(mesh_path)
    except Exception as exc:
        print(f"사물 위치 패널: CAD 로드 실패 ({exc})")
        if enabled:
            return None
        mesh_path = Path("cad")
        vertices_m = np.zeros((1, 3))
        faces = np.zeros((0, 3), dtype=np.uint32)
    try:
        place, grasp = load_specs(CAD_YAML)
    except Exception as exc:
        print(f"사물 위치 패널: yaml 로드 실패 ({exc})")
        return None
    return TeachGraspGui(
        parent,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
        kinematics=getattr(controller, "_kin", None) if controller is not None else None,
        controller=controller,
        own_robot=False,
        drive_viz=False,
        embedded=True,
        enabled=enabled,
        actions_parent=actions_parent,
    )


def attach_teach_window(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    geometry: str = "560x1200+1000+40",
) -> TeachGraspGui | None:
    """펜던트와 같은 Meshcat에 teach 창을 붙인다. 팔 display(q) 는 펜던트가 담당."""
    import customtkinter as ctk

    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_cad_mesh_m(mesh_path)
    except Exception as exc:
        print(f"teach 창 생략: CAD 로드 실패 ({exc})")
        return None
    place, grasp = load_specs(CAD_YAML)
    win = ctk.CTkToplevel(parent)
    win.geometry(geometry)
    win.minsize(520, 1100)
    gui = TeachGraspGui(
        win,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
        kinematics=getattr(controller, "_kin", None) if controller is not None else None,
        controller=controller,
        own_robot=False,
        drive_viz=False,
    )
    win.geometry(geometry)
    win.after(80, win.lift)
    return gui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="물체 배치·grasp 값을 Meshcat에서 확인·이동")
    p.add_argument("--no-open", action="store_true", help="Meshcat 브라우저를 자동으로 열지 않음")
    return p.parse_args()


def main() -> None:
    quiet_gtk()
    args = parse_args()
    try:
        import customtkinter as ctk  # noqa: F401
        import open3d as o3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "customtkinter / open3d 가 필요합니다. conda activate lerobot"
        ) from exc

    from motion import Controller, DEFAULT_URDF, RobotKinematics
    from visualizer import Visualizer

    mesh_path = cad_mesh_path()
    vertices_m, faces = load_cad_mesh_m(mesh_path)
    place, grasp = load_specs(CAD_YAML)

    kin = RobotKinematics(DEFAULT_URDF)
    viz = Visualizer(kin, open_browser=not args.no_open)
    url = viz.url or "(meshcat server running)"
    print(f"CAD      {mesh_path}")
    print(f"Meshcat  {url}")

    ctrl = Controller(kin)
    ctrl.start()

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    gui = TeachGraspGui(
        root,
        visualizer=viz,
        kinematics=kin,
        controller=ctrl,
        own_robot=False,
        drive_viz=True,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=url,
    )

    def _on_close() -> None:
        gui._cancel_register_poll()
        ctrl.stop()
        try:
            viz.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
