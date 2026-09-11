#!/usr/bin/env python3
"""물체 배치 + grasp yaml 값을 Meshcat에서 확인·티칭하는 GUI.

권장: 펜던트와 한 프로세스·한 Meshcat.

  python pendant/main.py              # 펜던트만
  python pendant/main.py --grasp      # 조그 + 물체 집기 창 (같은 Meshcat)
  python yolo/teach_grasp.py          # 물체 창만 (가상 팔로 P/G 이동)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PENDANT = PROJECT / "pendant"
for path in (PROJECT, PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolo.config import CAD_YAML, cad_mesh_path, cad_unit, quiet_gtk  # noqa: E402
from yolo.register import _to_mm  # noqa: E402

# 시뮬에서 물체를 책상에 올려 두는 기본값 (프로젝트 베이스 mm / deg).
# 프로젝트 베이스 = URDF Rz(180°). mesh_rpy Rx180 적용 후 place/grasp.
DEFAULT_PLACE_XYZ_MM = (-150.0, 150.0, 2.0)
DEFAULT_PLACE_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_GRASP_XYZ_MM = (0.0, 0.0, 0.0)
DEFAULT_GRASP_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_GRIPPER = 50.0
DEFAULT_APPROACH_D_MM = 10.0
DEFAULT_APPROACH_A_MM = 30.0  # pre → grasp along axis (mm toward CAD origin)
APPROACH_MARKER_RADIUS_M = 0.004  # 4 mm
# Teach →P / →G go-to (not jog). 30 mm/s, 45 deg/s.
GOTO_LIN_MPS = 0.030
GOTO_ROT_DEG_S = 45.0
# Random place (project base mm): polar sector ∩ box ∩ height, mesh above floor.
# x=r·cosθ, y=r·sinθ; θ ∈ (0°, 180°) → y > 0 half-plane.
RANDOM_XY_ABS_MAX_MM = 200.0
RANDOM_R_MIN_MM = 100.0  # r > 100 mm (10 cm)
RANDOM_R2_MIN_MM2 = RANDOM_R_MIN_MM * RANDOM_R_MIN_MM  # r² > 10000
RANDOM_THETA_MIN_DEG = 0.0
RANDOM_THETA_MAX_DEG = 180.0
RANDOM_Z_MIN_MM = 0.0
RANDOM_Z_MAX_MM = 40.0
RANDOM_PLACE_MAX_TRIES = 800
# Constrained random-place modes (project base).
STAND_Z_MM = 2.0
STAND_ROLL_DEG = 0.0
STAND_PITCH_DEG = 0.0
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
    form: str = ""  # "", "ㅅ", "V" when axis is N


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


def place_rpy_body_delta(
    rpy_deg: tuple[float, float, float] | np.ndarray,
    *,
    axis: str,
    delta_deg: float,
) -> tuple[float, float, float]:
    """Rotate about CAD (object) axes at the object origin, then re-express as base RPY.

    axis: ``roll``→CAD X, ``pitch``→CAD Y, ``yaw``→CAD Z.
    Returned numbers are extrinsic XYZ in project base (yaml / pose_to_T).
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
    return float(_wrap_deg(out[0])), float(_wrap_deg(out[1])), float(_wrap_deg(out[2]))


def frames_from_specs(place: PlacePose, grasp: GraspSpec) -> np.ndarray:
    """Return T_base_grasp in project base (URDF Rz180°)."""
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
    if err_p >= 0.01 or err_r >= 0.12:
        return None
    return s6


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
    """Pick approach marker; grasp = pre − a toward CAD origin."""
    T_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    R = T_cad[:3, :3]
    t = T_cad[:3, 3]
    base_z = np.array([0.0, 0.0, 1.0])

    # Candidates: object X/Y/Z and diagonal N (ㅅ/V).
    n_base = R @ N_CAD
    axis_dirs = (R[:, 0], R[:, 1], R[:, 2], n_base)
    axis_names = ("X", "Y", "Z", "N")
    dots = [abs(float(np.dot(a, base_z))) for a in axis_dirs]
    axis_i = int(np.argmax(dots))
    axis_name = axis_names[axis_i]

    pts_cad_mm = bracket_approach_points_cad_mm(mins_mm, maxs_mm, d_mm)
    # 0=+X 1=+Y 2=-Y 3=+Z 4=+N(ㅅ) 5=-N(V)
    if axis_name == "X":
        cand = [0]
    elif axis_name == "Y":
        cand = [1, 2]
    elif axis_name == "Z":
        cand = [3]
    else:
        cand = [4, 5]

    def _base_z_of(i: int) -> float:
        p = pts_cad_mm[i] / 1000.0
        return float((R @ p + t)[2])

    point_index = max(cand, key=_base_z_of)
    form = ""
    if axis_name == "N":
        form = "ㅅ" if point_index == 4 else "V"

    p_cad_mm = pts_cad_mm[point_index]
    nrm = float(np.linalg.norm(p_cad_mm))
    if nrm < 1e-9:
        raise ValueError("pre marker at CAD origin")
    g_cad_mm = p_cad_mm * (1.0 - float(a_mm) / nrm)
    p_base = R @ (p_cad_mm / 1000.0) + t
    g_base = R @ (g_cad_mm / 1000.0) + t

    if axis_name == "N":
        # Pitch ∥ N (sign = pre side). Roll ∥ ± object Y; min |ΔS6|.
        pitch_base = R @ (p_cad_mm / nrm)
        roll_cands = [R[:, 1], -R[:, 1]]
    elif axis_name == "X":
        pitch_base = R[:, 0]
        roll_cands = [R[:, 2], -R[:, 2]]
    elif axis_name == "Y":
        # +N only: −N puts the jaw into the bracket on pre→grasp descent.
        pitch_base = R[:, 1] if point_index == 1 else -R[:, 1]
        roll_cands = [n_base]
    else:  # Z
        pitch_base = R[:, 2]
        roll_cands = [R[:, 0], -R[:, 0]]

    R_tcp, flip = _pick_tcp_rot(
        pitch_base=pitch_base,
        roll_candidates=roll_cands,
        p_base=p_base,
        kin=kin,
        q=q,
        prefer_min_s6_delta=True,
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

    Face:
      +X (lx+d), ±Y (±(½ly+d)), +Z (lz+d)
    Diagonal N=(1,0,1)/√2 (ㅅ/V):
      +N·d , −N·d
    """
    mins = np.asarray(mins_mm, dtype=float).reshape(3)
    maxs = np.asarray(maxs_mm, dtype=float).reshape(3)
    lx, ly, lz = maxs - mins
    d = float(d_mm)
    n = N_CAD * d
    return np.asarray(
        [
            [lx + d, 0.0, 0.0],
            [0.0, 0.5 * ly + d, 0.0],
            [0.0, -(0.5 * ly + d), 0.0],
            [0.0, 0.0, lz + d],
            n,  # ㅅ (+N)
            -n,  # V (−N)
        ],
        dtype=float,
    )


def _theta_deg_xy(x: float, y: float) -> float:
    """atan2(y,x) in degrees for sector checks (0°, 180°) → y > 0.

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
    """Largest r with |r cosθ|<200 and |r sinθ|<200."""
    c = abs(float(np.cos(theta_rad)))
    s = abs(float(np.sin(theta_rad)))
    lim = np.inf
    if c > 1e-12:
        lim = min(lim, (RANDOM_XY_ABS_MAX_MM - 1e-3) / c)
    if s > 1e-12:
        lim = min(lim, (RANDOM_XY_ABS_MAX_MM - 1e-3) / s)
    return float(lim)


def _place_xy_in_region(x: float, y: float) -> bool:
    if abs(x) >= RANDOM_XY_ABS_MAX_MM or abs(y) >= RANDOM_XY_ABS_MAX_MM:
        return False
    if x * x + y * y <= RANDOM_R2_MIN_MM2:
        return False
    return _theta_in_sector(_theta_deg_xy(x, y))


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
    """One (x,y) in the S1 polar sector ∩ |x|,|y|<200, r > RANDOM_R_MIN_MM. None if impossible."""
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
    from yolo.register import apply_cad_mesh_frame

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise RuntimeError(f"삼각형 없음: {path}")
    verts_mm = apply_cad_mesh_frame(_to_mm(np.asarray(mesh.vertices), cad_unit(), path), source=path)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    return verts_mm / 1000.0, faces


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
    ) -> None:
        import customtkinter as ctk

        self.root = root
        self._viz = visualizer
        self._kin = kinematics
        self._ctrl = controller
        self._own_robot = bool(own_robot) and controller is None
        self._drive_viz = bool(drive_viz) and controller is not None
        self._yaml_path = CAD_YAML
        self._cad_loaded = False
        self._auto_pre: AutoPreResult | None = None
        self._status: ctk.CTkLabel | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.sliders: dict[str, ctk.CTkSlider] = {}
        self._rpy_sync = False
        self._rpy_apply_after: str | None = None
        self._replay_after: str | None = None
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
        # UI place rpy = CAD-axis body XYZ. _place_rpy_prev = base extrinsic for mesh/yaml.
        place_rpy_ui = rpy_extrinsic_to_body_xyz(place.rpy_deg)
        self._place_rpy_prev = tuple(float(x) for x in place.rpy_deg)
        self._place_slider_cmd = {
            "roll": float(place_rpy_ui[0]),
            "pitch": float(place_rpy_ui[1]),
            "yaw": float(place_rpy_ui[2]),
        }

        root.title("물체 집기 티칭")
        root.minsize(520, 900)
        root.geometry("560x960")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        if self._ctrl is not None:
            robot_hint = "→P / →G / P→G 로 가상·실물 팔을 목표로 보냄. 조그는 펜던트."
        elif self._own_robot:
            robot_hint = "단독 모드: 적용 시 팔은 HOME, 그리퍼만 반영. 이동은 펜던트 --grasp 권장."
        else:
            robot_hint = "팔 컨트롤러 없음. 오버레이만."
        ctk.CTkLabel(
            root,
            text=(
                f"CAD {mesh_path.name}  ·  Meshcat: {meshcat_url or 'printed URL'}\n"
                f"물체 xyz=베이스 · rpy 표시=물체(CAD) 축. grasp=CAD 축.\n"
                f"{robot_hint}"
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=14, pady=(14, 8))

        body = ctk.CTkScrollableFrame(root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=4)

        self._section(body, "시뮬 물체 (xyz=베이스 mm · rpy=물체축 deg)")
        self._xyzrpy_block(body, "place", place.xyz_mm, place_rpy_ui)
        ctk.CTkLabel(
            body,
            text="place rpy=물체 원점 CAD 축(R=Rx@Ry@Rz). xyz만 로봇 베이스.",
            text_color="#9ca3af",
            anchor="w",
        ).pack(fill="x", padx=8, pady=(0, 10))

        grasp_hdr = ctk.CTkFrame(body, fg_color="transparent")
        grasp_hdr.pack(fill="x", padx=8, pady=(12, 4))
        ctk.CTkLabel(
            grasp_hdr,
            text="grasp (CAD mm / deg)",
            anchor="w",
            font=ctk.CTkFont(size=15, weight="bold"),
        ).pack(side="left")
        self._grasp_hide_var = ctk.BooleanVar(value=False)
        self._grasp_hide_cb = ctk.CTkCheckBox(
            grasp_hdr,
            text="비활성화 (축 숨김)",
            variable=self._grasp_hide_var,
            command=self._on_grasp_hide_toggle,
            width=160,
        )
        self._grasp_hide_cb.pack(side="right")
        self._xyzrpy_block(body, "grasp", grasp.xyz_mm, grasp.rpy_deg)
        self._row(body, "gripper", f"{grasp.gripper:.1f}", "벌림 0–100")
        grip_note = (
            "→P = pre(주황점), →G = grasp(a 진입). P→G 는 아직 미구현."
            if self._ctrl is not None
            else "단독 모드에서만 적용 시 턱 벌림에 반영."
        )
        ctk.CTkLabel(
            body,
            text=(
                "주황=선택 pre. 초록=grasp(a). N=(1,0,1): ㅅ=+N·d, V=−N·d.\n"
                "축 후보 X/Y/Z/N. pitch∥선택축(가능하면 flip 없음), roll∥짝축 ± 중 |ΔS6| 최소(±165°; Y축은 +N만).\n"
                + grip_note
            ),
            text_color="#9ca3af",
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 10))
        self._grasp_ui_keys = (
            "grasp_x",
            "grasp_y",
            "grasp_z",
            "grasp_roll",
            "grasp_pitch",
            "grasp_yaw",
            "gripper",
        )

        bar = ctk.CTkFrame(root, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(4, 4))
        ctk.CTkButton(bar, text="적용", width=80, command=self.apply).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="물체 초기화", width=100, command=self.reset_place).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(bar, text="grasp 초기화", width=110, command=self.reset_grasp).pack(
            side="left", padx=(0, 6)
        )

        move = ctk.CTkFrame(root, fg_color="transparent")
        move.pack(fill="x", padx=14, pady=(0, 8))
        self._btn_pre = ctk.CTkButton(move, text="→ P", width=90, command=self.goto_pre)
        self._btn_grasp = ctk.CTkButton(move, text="→ G", width=90, command=self.goto_grasp)
        self._btn_seq = ctk.CTkButton(move, text="P → G", width=90, command=self.replay_pg)
        self._btn_init = ctk.CTkButton(
            move, text="초기자세로 이동", width=130, command=self.snap_init_pose
        )
        self._btn_pre.pack(side="left", padx=(0, 6))
        self._btn_grasp.pack(side="left", padx=(0, 6))
        self._btn_seq.pack(side="left", padx=(0, 6))
        self._btn_init.pack(side="left")
        # P→G 시퀀스는 아직. →P / →G 는 자동 pre·grasp.
        self._btn_seq.configure(state="disabled")
        if self._ctrl is None:
            self._btn_pre.configure(state="disabled")
            self._btn_grasp.configure(state="disabled")
            self._btn_init.configure(state="disabled")

        rand_row = ctk.CTkFrame(root, fg_color="transparent")
        rand_row.pack(fill="x", padx=14, pady=(0, 2))
        ctk.CTkButton(
            rand_row,
            text="물체 랜덤 생성",
            width=140,
            command=self.randomize_place,
        ).pack(side="left")
        self._warn_label = ctk.CTkLabel(
            rand_row,
            text="",
            anchor="w",
            text_color="#f87171",
        )
        self._warn_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        mode_row = ctk.CTkFrame(root, fg_color="transparent")
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
            cb.pack(side="left", padx=(0, 14))
            self._rand_mode_vars[key] = var

        foot = ctk.CTkFrame(root, fg_color="transparent")
        foot.pack(fill="x", padx=14, pady=(0, 4))
        lx, ly, lz = self._cad_maxs_mm - self._cad_mins_mm
        ctk.CTkLabel(
            foot,
            text=(
                f"pre=L+d 또는 ±N·d · grasp=pre−a  "
                f"lx={lx:.1f} ly={ly:.1f} lz={lz:.1f} mm"
            ),
            anchor="w",
            text_color="#9ca3af",
        ).pack(fill="x", pady=(0, 2))
        self._row(foot, "approach_d", f"{approach_d0:.1f}", "d mm (pre)")
        self._row(foot, "approach_a", f"{approach_a0:.1f}", "a mm (grasp)")
        self.entries["approach_d"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_d"].bind("<Return>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<Return>", lambda _e: self._schedule_live_apply())

        self._status = ctk.CTkLabel(root, text="", anchor="w")
        self._status.pack(fill="x", padx=14, pady=(0, 14))

        root.bind("<Return>", lambda _e: self.apply())
        self._draw_static(vertices_m, faces)
        self.apply()
        if self._drive_viz:
            self._schedule_display()

    def _section(self, parent, title: str) -> None:
        import customtkinter as ctk

        ctk.CTkLabel(parent, text=title, anchor="w", font=ctk.CTkFont(size=15, weight="bold")).pack(
            fill="x", padx=8, pady=(12, 4)
        )

    def _grasp_section_hidden(self) -> bool:
        var = getattr(self, "_grasp_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_grasp_hide_toggle(self) -> None:
        hidden = self._grasp_section_hidden()
        state = "disabled" if hidden else "normal"
        for key in getattr(self, "_grasp_ui_keys", ()):
            if key in self.entries:
                self.entries[key].configure(state=state)
            if key in self.sliders:
                self.sliders[key].configure(state=state)
        self.apply()

    def _hide_grasp_overlays(self) -> None:
        for name in ("grasp", "grasp_marker", "approach"):
            self._viz.clear_overlay(name)

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
            (f"{prefix}_roll", "roll deg", rpy[0]),
            (f"{prefix}_pitch", "pitch deg", rpy[1]),
            (f"{prefix}_yaw", "yaw deg", rpy[2]),
        ):
            self._rpy_row(parent, key, float(val), label)

    def _row(self, parent, key: str, value: str, label: str) -> None:
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
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
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        ent = ctk.CTkEntry(row, width=70)
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

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        ent = ctk.CTkEntry(row, width=70)
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

    def _nudge_place_rpy(self, axis: str, new_axis_val: float) -> None:
        """Set one CAD-axis body-XYZ angle; rebuild base extrinsic for mesh/yaml."""
        self._place_slider_cmd[axis] = float(new_axis_val)
        self._place_rpy_prev = rpy_body_xyz_to_extrinsic(
            self._place_slider_cmd["roll"],
            self._place_slider_cmd["pitch"],
            self._place_slider_cmd["yaw"],
        )
        self._schedule_live_apply()

    def _on_rpy_slider(self, key: str, value: float) -> None:
        if self._rpy_sync:
            return
        if key.startswith("place_"):
            self._rpy_sync = True
            try:
                ent = self.entries[key]
                ent.delete(0, "end")
                ent.insert(0, f"{float(value):.1f}")
            finally:
                self._rpy_sync = False
            self._nudge_place_rpy(key.removeprefix("place_"), float(value))
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
            axis = key.removeprefix("place_")
            self._nudge_place_rpy(axis, val)
            self._rpy_sync = True
            try:
                self.sliders[key].set(val)
                self.entries[key].delete(0, "end")
                self.entries[key].insert(0, f"{val:.1f}")
            finally:
                self._rpy_sync = False
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

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if self._status is None:
            return
        self._status.configure(text=text, text_color="#f87171" if error else "#86efac")

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
        gripper = self._read_float("gripper")
        if not 0.0 <= gripper <= 100.0:
            raise ValueError("gripper 는 0–100")
        grasp = GraspSpec(
            xyz_mm=(
                self._read_float("grasp_x"),
                self._read_float("grasp_y"),
                self._read_float("grasp_z"),
            ),
            rpy_deg=(
                self._read_float("grasp_roll"),
                self._read_float("grasp_pitch"),
                self._read_float("grasp_yaw"),
            ),
            gripper=gripper,
        )
        return place, grasp

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
        # Mesh/yaml = base extrinsic. Pendant shows CAD-axis body XYZ of that pose.
        self._place_rpy_prev = tuple(float(x) for x in place.rpy_deg)
        body = rpy_extrinsic_to_body_xyz(place.rpy_deg)
        self._place_slider_cmd = {
            "roll": float(body[0]),
            "pitch": float(body[1]),
            "yaw": float(body[2]),
        }
        mapping = {
            "place_x": place.xyz_mm[0],
            "place_y": place.xyz_mm[1],
            "place_z": place.xyz_mm[2],
            "place_roll": body[0],
            "place_pitch": body[1],
            "place_yaw": body[2],
        }
        for key, val in mapping.items():
            self._set_entry(key, val)

    def _fill_grasp(self, grasp: GraspSpec) -> None:
        mapping = {
            "grasp_x": grasp.xyz_mm[0],
            "grasp_y": grasp.xyz_mm[1],
            "grasp_z": grasp.xyz_mm[2],
            "grasp_roll": grasp.rpy_deg[0],
            "grasp_pitch": grasp.rpy_deg[1],
            "grasp_yaw": grasp.rpy_deg[2],
            "gripper": grasp.gripper,
        }
        for key, val in mapping.items():
            self._set_entry(key, val)

    def _draw_static(self, vertices_m: np.ndarray, faces: np.ndarray) -> None:
        table_t = 0.004
        T_table = np.eye(4)
        T_table[2, 3] = -table_t / 2.0
        self._viz.set_overlay_box(
            "table",
            (0.45, 0.45, table_t),
            T_table,
            color=0x2F2F35,
            opacity=0.45,
        )
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

    def _schedule_display(self) -> None:
        if self._ctrl is None:
            return
        st = self._ctrl.snapshot()
        self._viz.display(st.q)
        self.root.after(DISPLAY_MS, self._schedule_display)

    def _read_da(self) -> tuple[float, float]:
        try:
            d_mm = self._read_float("approach_d")
        except (KeyError, ValueError):
            d_mm = DEFAULT_APPROACH_D_MM
        try:
            a_mm = self._read_float("approach_a")
        except (KeyError, ValueError):
            a_mm = DEFAULT_APPROACH_A_MM
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

    def apply(self) -> None:
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
        self._viz.set_overlay_axes("cad_axes", T_cad, scale=0.03, tag="C")
        self._viz.set_overlay_axes("pregrasp", T_p, scale=0.035, tag="P")
        # Virtual N axis (magenta) through CAD origin.
        n_world = T_cad[:3, :3] @ N_CAD
        o = T_cad[:3, 3]
        half = max(float(d_mm), 20.0) / 1000.0
        self._viz.set_overlay_segment(
            "n_axis",
            o - n_world * half,
            o + n_world * half,
            color=0xC026D3,
        )
        pts_cad_m = bracket_approach_points_cad_mm(self._cad_mins_mm, self._cad_maxs_mm, d_mm) / 1000.0
        pts_world = (T_cad[:3, :3] @ pts_cad_m.T).T + T_cad[:3, 3]
        colors = [0x1E88E8] * len(pts_world)
        colors[auto.point_index] = 0xFB923C  # selected pre
        self._viz.set_overlay_spheres(
            "approach_markers",
            pts_world,
            radius_m=APPROACH_MARKER_RADIUS_M,
            colors=colors,
        )
        if self._grasp_section_hidden():
            self._hide_grasp_overlays()
        else:
            self._viz.set_overlay_axes("grasp", T_g, scale=0.05, tag="G")
            self._viz.set_overlay_segment("approach", T_p[:3, 3], T_g[:3, 3])
            self._viz.set_overlay_spheres(
                "grasp_marker",
                T_g[:3, 3].reshape(1, 3),
                radius_m=APPROACH_MARKER_RADIUS_M,
                color=0x22C55E,
            )
        if self._own_robot:
            self._show_robot(grasp.gripper)
        flip_txt = "pitch−" if auto.pitch_flipped else "pitch+"
        form_txt = f" {auto.form}" if auto.form else ""
        hide_txt = "  G숨김" if self._grasp_section_hidden() else ""
        self._set_status(
            f"적용  CAD xyz={list(np.round(place.xyz_mm, 1))}  "
            f"axis={auto.axis_name}{form_txt} {flip_txt}  d={d_mm:.1f} a={a_mm:.1f}  "
            f"gripper={grasp.gripper:.1f}{hide_txt}"
        )

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
        duration_s = self._ctrl.start_ee_goto(
            xyz_mm,
            rpy_deg,
            lin_mps=GOTO_LIN_MPS,
            rot_deg_s=GOTO_ROT_DEG_S,
        )
        self._set_status(
            f"{label} 이동  xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
            f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]  "
            f"({duration_s:.1f}s @ {GOTO_LIN_MPS*1000:.0f}mm/s, {GOTO_ROT_DEG_S:.0f}°/s)"
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
        self._goto_T(auto.T_base, label=f"→ P ({auto.axis_name}{form_txt} {flip_txt})")

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
            label=f"→ G ({auto.axis_name}{form_txt} a={auto.a_mm:.0f} {flip_txt})",
        )

    def replay_pg(self) -> None:
        self._set_status("P → G 는 아직 미구현", error=True)

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
        if self._replay_after is not None:
            try:
                self.root.after_cancel(self._replay_after)
            except Exception:
                pass
            self._replay_after = None
        self.root.destroy()


def attach_teach_window(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    geometry: str = "500x880+1000+40",
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
    TeachGraspGui(
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
        ctrl.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
