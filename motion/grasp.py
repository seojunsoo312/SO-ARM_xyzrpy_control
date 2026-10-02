"""물체 자세에서 집기 좌표를 만든다. GUI, 카메라 없음.

물체 배치, 물체 축 RPY와 베이스 RPY 변환, 자동 대기(P)·집기(G) TCP, 랜덤 배치, 물체-팔 겹침.
좌표는 프로젝트 베이스(+X 앞, +Y 왼쪽, +Z 위), mm와 deg. 4x4 T 의 위치는 m.
펜던트 물체 칸(pendant/teach_grasp.py)과 load_pick_targets()가 같이 쓴다.

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

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pinocchio as pin

from cad.model import (
    CAD_YAML,
    DEFAULT_APPROACH_A_MM,
    grasp_xyzrpy_gripper,
    place_xyzrpy,
)
from motion.base_frame import R_URDF_FROM_USER, T_urdf_from_user, ee_target_urdf_from_user
from motion.robot_kinematics import (
    JOINT_LIMIT_USER_DEG,
    rotmat_to_rpy_deg,
    rpy_deg_to_rotmat,
)

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


def load_specs(path: Path = CAD_YAML) -> tuple[PlacePose, GraspSpec]:
    """model.yaml `place`, `grasp` → dataclass."""
    place_xyz, place_rpy = place_xyzrpy(path)
    grasp_xyz, grasp_rpy, gripper = grasp_xyzrpy_gripper(path)
    return (
        PlacePose(xyz_mm=place_xyz, rpy_deg=place_rpy),
        GraspSpec(xyz_mm=grasp_xyz, rpy_deg=grasp_rpy, gripper=gripper),
    )


def pose_to_T(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    T[:3, 3] = np.asarray(xyz_mm, dtype=float).reshape(3) / 1000.0
    return T


def T_to_xyzrpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(T, dtype=float)
    xyz_mm = M[:3, 3] * 1000.0
    rpy_deg = rotmat_to_rpy_deg(M[:3, :3])
    return xyz_mm, rpy_deg


def _wrap_deg(deg: float) -> float:
    """Map degrees into (-180, 180]."""
    return float((float(deg) + 180.0) % 360.0 - 180.0)


def rpy_body_xyz_to_rotmat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """CAD-axis (object) RPY: R = Rx(roll) @ Ry(pitch) @ Rz(yaw)."""
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
    r = np.asarray(rpy_deg, dtype=float).reshape(3)
    return rotmat_to_rpy_body_xyz(rpy_deg_to_rotmat(float(r[0]), float(r[1]), float(r[2])))


def rpy_body_xyz_to_extrinsic(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float]:
    """Pendant CAD-axis RPY → base extrinsic for pose_to_T / yaml."""
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


def place_region_overlay_m(n_arc: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random-place XY region as a thin sheet in URDF meters.

    Returns (verts, faces, outline_xyz). Outline is (N, 3) along the outer then
    reversed inner boundary.
    """
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
    u = np.asarray(u_cad, dtype=float).reshape(3)
    n = float(np.linalg.norm(u))
    if n < 1e-12:
        raise ValueError("zero axis")
    u = u / n
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    return np.asarray(pin.Quaternion.FromTwoVectors(u, ez).toRotationMatrix(), dtype=float)


def _rpy_slant_about_s(phi_deg: float) -> tuple[float, float, float]:
    """R = Rz(φ) @ R0, R0 maps SLANT_AXIS_CAD → +Z (ㅅ, free spin about that axis)."""
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
