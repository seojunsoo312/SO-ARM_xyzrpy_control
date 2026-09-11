"""Virtual / real jog / go-to. Soft joint limits via clamp_q; mesh collision blocks the step."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
import pinocchio as pin

from motion.base_frame import (
    R_URDF_FROM_USER,
    ee_target_urdf_from_user,
    tcp_pose_user_from_urdf,
    xyz_user_from_urdf,
)
from motion.hw_controller import Hardware
from motion.pose_server import POSE_URL, PoseServer
from motion.robot_kinematics import (
    DEG2RAD,
    GRIPPER_CLOSED_CAD_DEG,
    GRIPPER_JOINT,
    GRIPPER_OPEN_CAD_DEG,
    HOME_JOINTS_DEG,
    INIT_POSE_JOINTS_DEG,
    JOINT_SIGN,
    TCP_FRAME,
    URDF_JOINT_NAMES,
    RobotKinematics,
    TcpPose,
)

FPS = 30
JOINT_VEL_DEG_S = 20.0
GRIPPER_VEL_UNIT_S = 40.0
JOG_VEL_MPS = 0.010  # default XYZ jog; GUI slider 5–15 mm/s
JOG_VEL_MIN_MPS = 0.005
JOG_VEL_MAX_MPS = 0.015
JOG_ROT_DEG_S = 20.0
JOG_ROT_MIN_DEG_S = 5.0
JOG_ROT_MAX_DEG_S = 45.0
JOG_ROT_RAD_S = JOG_ROT_DEG_S * DEG2RAD
TORQUE_PCT_MIN = 10.0
TORQUE_PCT_MAX = 100.0
DEFAULT_ARM_TORQUE_PCT = 100.0
DEFAULT_GRIPPER_TORQUE_PCT = 50.0
MAX_TARGET_LEAD_M = 0.012  # 12 mm vs measured TCP (real). Sub-tick goals never move STS3215.
# q_send = (1-β) q_prev + β q_ik. Smaller β = less shake, more lag.
CART_IK_BLEND = 0.28
GOTO_DURATION_S = 2.5
GOTO_MIN_S = 0.2
GOTO_MAX_S = 30.0
Z_FLOOR_M = 0.0

CART_AXES = ("x", "y", "z", "wx", "wy", "wz")
ROT_FRAME_BASE = "base"
ROT_FRAME_TCP = "tcp"


@dataclass(frozen=True)
class PendantState:
    q: np.ndarray
    joints_deg: dict[str, float]
    pose: TcpPose
    fault: str
    mode: str
    rot_frame: str
    connected: bool
    torque: bool
    ee_err_mm: float | None
    err_xyz_mm: np.ndarray | None


class Controller:
    def __init__(self, kinematics: RobotKinematics, hardware: Hardware | None = None) -> None:
        self._kin = kinematics
        self._hw = hardware
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._joint_jog = dict.fromkeys(URDF_JOINT_NAMES, 0)
        self._cart_jog = dict.fromkeys(CART_AXES, 0)
        self._q = kinematics.q_home()
        self._pose = kinematics.forward_tcp(self._q)
        self._target_xyz: np.ndarray | None = None
        self._target_rot: np.ndarray | None = None
        self._fault = ""
        self._goto: dict | None = None
        self._mode = "virtual"
        self._rot_frame = ROT_FRAME_BASE
        self._q_meas: np.ndarray | None = None
        self._pose_meas: TcpPose | None = None
        self._ee_err_mm: float | None = None
        self._err_xyz_mm: np.ndarray | None = None
        self._hw_hold = False
        self._pose_server: PoseServer | None = None
        self._jog_vel_mps = float(JOG_VEL_MPS)
        self._jog_rot_rad_s = float(JOG_ROT_RAD_S)
        self._torque_pct = {
            name: (
                DEFAULT_GRIPPER_TORQUE_PCT
                if name == GRIPPER_JOINT
                else DEFAULT_ARM_TORQUE_PCT
            )
            for name in URDF_JOINT_NAMES
        }

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        if self._pose_server is None:
            self._pose_server = PoseServer(self.handeye_payload)
            try:
                self._pose_server.start()
                print(f"hand-eye pose  {POSE_URL}")
            except OSError as exc:
                self._pose_server = None
                print(f"hand-eye pose server failed ({exc})")

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._pose_server is not None:
            self._pose_server.stop()
            self._pose_server = None
        self.disconnect()

    def set_joint_jog(self, joint: str, sign: int) -> None:
        if joint not in self._joint_jog:
            raise KeyError(joint)
        with self._lock:
            if sign != 0 and self._frozen_unlocked():
                return
            self._goto = None
            self._joint_jog[joint] = int(np.sign(sign))
            if sign != 0:
                self._reset_cart_targets()

    def set_tcp_jog_mm_s(self, mm_s: float) -> None:
        lo = JOG_VEL_MIN_MPS * 1000.0
        hi = JOG_VEL_MAX_MPS * 1000.0
        vel = float(np.clip(mm_s, lo, hi)) / 1000.0
        with self._lock:
            self._jog_vel_mps = vel

    def tcp_jog_mm_s(self) -> float:
        with self._lock:
            return self._jog_vel_mps * 1000.0

    def set_rpy_jog_deg_s(self, deg_s: float) -> None:
        vel = float(np.clip(deg_s, JOG_ROT_MIN_DEG_S, JOG_ROT_MAX_DEG_S))
        with self._lock:
            self._jog_rot_rad_s = vel * DEG2RAD

    def rpy_jog_deg_s(self) -> float:
        with self._lock:
            return self._jog_rot_rad_s / DEG2RAD

    def set_joint_torque_pct(self, joint: str, pct: float) -> None:
        if joint not in self._torque_pct:
            raise KeyError(joint)
        value = float(np.clip(pct, TORQUE_PCT_MIN, TORQUE_PCT_MAX))
        with self._lock:
            self._torque_pct[joint] = value
        hw = self._hw
        if hw is None or not hw.is_connected or not hw.torque_enabled:
            return
        try:
            hw.set_torque_pct(joint, value)
        except Exception as exc:
            with self._lock:
                self._fault = f"torque write failed ({exc})"

    def joint_torque_pct(self, joint: str) -> float:
        with self._lock:
            return float(self._torque_pct[joint])

    def _apply_torque_pcts_to_hw(self) -> None:
        hw = self._hw
        if hw is None or not hw.is_connected:
            return
        with self._lock:
            pcts = dict(self._torque_pct)
        try:
            hw.apply_torque_pcts(pcts)
        except Exception as exc:
            raise RuntimeError(f"torque limit write failed ({exc})") from exc

    def set_rot_frame(self, frame: str) -> None:
        """RPY jog frame: 'base' (world-fixed) or 'tcp' (tool-local)."""
        key = str(frame).strip().lower()
        if key in {"base", "world"}:
            key = ROT_FRAME_BASE
        elif key in {"tcp", "tool", "local"}:
            key = ROT_FRAME_TCP
        else:
            raise ValueError(f"rot frame must be base|tcp, got {frame!r}")
        with self._lock:
            self._rot_frame = key
            self._reset_cart_targets()

    def rot_frame(self) -> str:
        with self._lock:
            return self._rot_frame

    def set_cart_jog(self, axis: str, sign: int) -> None:
        if axis not in self._cart_jog:
            raise KeyError(axis)
        with self._lock:
            if sign != 0 and self._frozen_unlocked():
                return
            self._goto = None
            self._cart_jog[axis] = int(np.sign(sign))
            if not any(self._cart_jog.values()):
                self._reset_cart_targets()

    def clear_jog(self) -> None:
        """Stop hold-to-jog only. Absolute go-to keeps running."""
        with self._lock:
            for name in self._joint_jog:
                self._joint_jog[name] = 0
            for name in self._cart_jog:
                self._cart_jog[name] = 0
            self._reset_cart_targets()

    def stop_all(self) -> None:
        with self._lock:
            self._clear_motion_unlocked()
            self._fault = ""

    def home(self) -> None:
        self.start_joint_goto(dict(HOME_JOINTS_DEG))

    def init_pose(self) -> None:
        """작업 시작 자세 (펜던트 초기자세 버튼)."""
        self.start_joint_goto(dict(INIT_POSE_JOINTS_DEG))

    def snap_to_joints(self, joints_deg: dict[str, float]) -> bool:
        """관절을 보간 없이 목표로 즉시 설정 (가상 전용; 실기 write 경로도 타지만 호출부에서 막음)."""
        with self._lock:
            if self._frozen_unlocked():
                return False
            self._clear_motion_unlocked()
        q = self._kin.q_from_deg(joints_deg)
        return self._commit(q)

    def snap_init_pose(self) -> str:
        """티칭 '초기자세로 이동': 가상은 즉시 스냅, 실기 연결 시 관절 go-to.

        Returns:
            ``"snap"`` | ``"goto"`` | ``""`` (실패/동결).
        """
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        with self._lock:
            mode = self._mode
            frozen = self._frozen_unlocked()
        if frozen:
            return ""
        if connected and mode == "real":
            self.init_pose()
            return "goto"
        if not self.snap_to_joints(dict(INIT_POSE_JOINTS_DEG)):
            return ""
        return "snap"

    def connect(self) -> None:
        if self._hw is None:
            raise RuntimeError("no hardware configured")
        with self._lock:
            self._clear_motion_unlocked()
            self._hw_hold = True
        try:
            q = self._hw.connect(self._kin)
            pose = self._kin.forward_tcp(q)
            with self._lock:
                self._q = q
                self._pose = pose
                self._q_meas = q.copy()
                self._pose_meas = pose
                self._ee_err_mm = 0.0
                self._err_xyz_mm = np.zeros(3)
                self._mode = "real"
                self._hw_hold = False
                self._fault = ""
            self._apply_torque_pcts_to_hw()
        except Exception:
            with self._lock:
                self._hw_hold = False
                self._mode = "virtual"
            if self._hw is not None and self._hw.is_connected:
                self._hw.disconnect()
            raise

    def disconnect(self) -> None:
        hw = self._hw
        if hw is not None and hw.is_connected:
            hw.disconnect()
        with self._lock:
            self._clear_motion_unlocked()
            if self._q_meas is not None:
                self._q = self._q_meas.copy()
                if self._pose_meas is not None:
                    self._pose = self._pose_meas
            self._mode = "virtual"
            self._q_meas = None
            self._pose_meas = None
            self._ee_err_mm = None
            self._err_xyz_mm = None
            self._fault = ""
            self._hw_hold = False

    def estop(self, reason: str = "E-stop — torque off") -> None:
        with self._lock:
            self._clear_motion_unlocked()
            self._fault = reason
        if self._hw is not None and self._hw.is_connected:
            try:
                self._hw.disable_torque()
            except Exception as exc:
                with self._lock:
                    self._fault = f"{reason} ({exc})"

    def start_ee_goto(
        self,
        xyz_mm: np.ndarray,
        rpy_deg: np.ndarray,
        *,
        duration_s: float | None = None,
        lin_mps: float | None = None,
        rot_deg_s: float | None = None,
    ) -> float:
        """Go-to TCP. xyz/rpy are project base (URDF Rz180°), not raw URDF.

        기본 시간은 XYZ/RPY 조그 속도. duration_s 를 주면 그 시간을 쓴다.
        lin_mps / rot_deg_s 를 주면 조그 속도 대신 그 값으로 시간을 잡는다.
        """
        xyz1, R1 = ee_target_urdf_from_user(xyz_mm, rpy_deg)
        with self._lock:
            if self._frozen_unlocked():
                return 0.0
            self._clear_motion_unlocked()
            xyz0 = self._pose.xyz_m.copy()
            R0 = self._pose.rotation.copy()
            if duration_s is None:
                duration_s = self._ee_goto_duration_unlocked(
                    xyz0,
                    R0,
                    xyz1,
                    R1,
                    lin_mps=lin_mps,
                    rot_deg_s=rot_deg_s,
                )
            duration_s = float(np.clip(duration_s, GOTO_MIN_S, GOTO_MAX_S))
            self._goto = {
                "kind": "ee",
                "i": 0,
                "n": max(int(duration_s * FPS), 1),
                "xyz0": xyz0,
                "R0": R0,
                "xyz1": xyz1,
                "R1": R1,
            }
            self._fault = ""
        return duration_s

    def start_joint_goto(
        self,
        joints_deg: dict[str, float],
        *,
        duration_s: float | None = None,
    ) -> float:
        """관절 go-to. 기본 시간은 관절별 토크% × 조그 각속도."""
        with self._lock:
            if self._frozen_unlocked():
                return 0.0
            self._clear_motion_unlocked()
            q1 = self._kin.q_from_deg(joints_deg)
            q0 = self._q.copy()
            if float(np.max(np.abs(q1 - q0))) < 1e-6:
                self._goto = None
                self._fault = "already at joint target"
                return 0.0
            if duration_s is None:
                duration_s = self._joint_goto_duration_unlocked(q0, q1)
            duration_s = float(np.clip(duration_s, GOTO_MIN_S, GOTO_MAX_S))
            self._goto = {
                "kind": "joint",
                "i": 0,
                "n": max(int(duration_s * FPS), 1),
                "q0": q0,
                "q1": q1,
            }
            self._fault = ""
        return duration_s

    def _ee_goto_duration_unlocked(
        self,
        xyz0: np.ndarray,
        R0: np.ndarray,
        xyz1: np.ndarray,
        R1: np.ndarray,
        *,
        lin_mps: float | None = None,
        rot_deg_s: float | None = None,
    ) -> float:
        lin = float(self._jog_vel_mps if lin_mps is None else lin_mps)
        rot = float(self._jog_rot_rad_s if rot_deg_s is None else (rot_deg_s * DEG2RAD))
        dist = float(np.linalg.norm(np.asarray(xyz1, dtype=float) - np.asarray(xyz0, dtype=float)))
        t_lin = dist / max(lin, 1e-6)
        ang = float(np.linalg.norm(pin.log3(np.asarray(R0, dtype=float).T @ np.asarray(R1, dtype=float))))
        t_rot = ang / max(rot, 1e-9)
        return max(t_lin, t_rot, GOTO_MIN_S)

    def _joint_goto_duration_unlocked(self, q0: np.ndarray, q1: np.ndarray) -> float:
        t_max = GOTO_MIN_S
        grip_span = abs(GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG)
        for name in URDF_JOINT_NAMES:
            i = self._kin.q_index(name)
            dq_deg = abs(float(q1[i] - q0[i])) / DEG2RAD
            if dq_deg < 1e-6:
                continue
            if name == GRIPPER_JOINT:
                vel_deg_s = (grip_span / 100.0) * GRIPPER_VEL_UNIT_S
            else:
                vel_deg_s = JOINT_VEL_DEG_S
            scale = float(self._torque_pct[name]) / 100.0
            vel = max(vel_deg_s * scale, 1e-3)
            t_max = max(t_max, dq_deg / vel)
        return t_max

    def snapshot(self) -> PendantState:
        with self._lock:
            q = self._q.copy()
            pose = tcp_pose_user_from_urdf(self._pose)
            fault = self._fault
            mode = self._mode
            rot_frame = self._rot_frame
            err = self._ee_err_mm
            err_xyz = None if self._err_xyz_mm is None else xyz_user_from_urdf(self._err_xyz_mm)
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        torque = bool(hw is not None and hw.torque_enabled)
        return PendantState(
            q=q,
            joints_deg=self._kin.joints_deg(q),
            pose=pose,
            fault=fault,
            mode=mode,
            rot_frame=rot_frame,
            connected=connected,
            torque=torque,
            ee_err_mm=err,
            err_xyz_mm=err_xyz,
        )

    def handeye_payload(self) -> dict:
        """Measured TCP in project base (Rz180°). Fallback = commanded pose."""
        with self._lock:
            q = self._q_meas.copy() if self._q_meas is not None else self._q.copy()
            pose_u = self._pose_meas if self._pose_meas is not None else self._pose
            pose = tcp_pose_user_from_urdf(pose_u)
            mode = self._mode
            fault = self._fault
            # Same frame as pendant err label: cmd − meas, project base mm.
            ee_err = self._ee_err_mm
            err_xyz = None if self._err_xyz_mm is None else xyz_user_from_urdf(self._err_xyz_mm)
        hw = self._hw
        connected = bool(hw is not None and hw.is_connected)
        torque = bool(hw is not None and hw.torque_enabled)
        T = np.eye(4)
        T[:3, :3] = np.asarray(pose.rotation, dtype=float)
        T[:3, 3] = np.asarray(pose.xyz_mm, dtype=float)
        return {
            "ok": bool(connected and torque and mode == "real"),
            "connected": connected,
            "torque": torque,
            "mode": mode,
            "fault": fault,
            "gripper_frame": TCP_FRAME,
            "base_frame": "project_rz180",
            "joints_deg": {k: float(v) for k, v in self._kin.joints_deg(q).items()},
            "tcp_xyz_mm": np.asarray(pose.xyz_mm, dtype=float).tolist(),
            "tcp_rpy_deg": np.asarray(pose.rpy_deg, dtype=float).tolist(),
            "T_base_tcp": T.tolist(),
            "ee_err_mm": None if ee_err is None else float(ee_err),
            "err_xyz_mm": None if err_xyz is None else np.asarray(err_xyz, dtype=float).tolist(),
        }

    def _frozen_unlocked(self) -> bool:
        """True while connecting, or after E-stop (bus up, torque off)."""
        if self._hw_hold:
            return True
        hw = self._hw
        return bool(self._mode == "real" and hw is not None and hw.is_connected and not hw.torque_enabled)

    def _clear_motion_unlocked(self) -> None:
        for name in self._joint_jog:
            self._joint_jog[name] = 0
        for name in self._cart_jog:
            self._cart_jog[name] = 0
        self._reset_cart_targets()
        self._goto = None

    def _reset_cart_targets(self) -> None:
        self._target_xyz = None
        self._target_rot = None

    def _commit(self, q: np.ndarray, *, rewind_cart: bool = False, allow_lift_escape: bool = True) -> bool:
        with self._lock:
            frozen = self._frozen_unlocked()
            q_now = self._q.copy()
            z_now = float(self._pose.xyz_m[2])
        if frozen:
            return False
        q = self._kin.clamp_q(q)
        pose = self._kin.forward_tcp(q)
        # Already-folded poses overlap L1/L2 vs L4. Jog may escape with a real +Z
        # step even if that newly collides. Go-to never uses that exception —
        # otherwise EE paths can tunnel gripper/wrist through the column.
        entering = self._kin.in_collision(q) and not self._kin.in_collision(q_now)
        lifting = float(pose.xyz_m[2]) >= z_now + 1e-4
        if entering and not (allow_lift_escape and lifting):
            with self._lock:
                self._fault = "collision — step ignored"
                if rewind_cart:
                    self._reset_cart_targets()
            return False
        if pose.xyz_m[2] < Z_FLOOR_M and z_now >= Z_FLOOR_M:
            with self._lock:
                self._fault = "floor — step ignored"
                if rewind_cart:
                    self._reset_cart_targets()
            return False
        with self._lock:
            self._q = q
            self._pose = pose
            self._fault = ""
        hw = self._hw
        if hw is not None and hw.torque_enabled:
            try:
                hw.write_q(self._kin, q)
            except Exception as exc:
                self.estop(f"E-stop — send failed ({exc})")
                return False
        return True

    def _refresh_meas(self) -> None:
        hw = self._hw
        with self._lock:
            mode = self._mode
            hold = self._hw_hold
        if hold or mode != "real" or hw is None or not hw.is_connected:
            return
        try:
            q_meas = hw.read_q(self._kin)
        except Exception as exc:
            self.estop(f"E-stop — read failed ({exc})")
            return
        pose_meas = self._kin.forward_tcp(q_meas)
        with self._lock:
            self._q_meas = q_meas
            self._pose_meas = pose_meas
            delta = (self._pose.xyz_m - pose_meas.xyz_m) * 1000.0
            self._err_xyz_mm = delta
            self._ee_err_mm = float(np.linalg.norm(delta))

    def _loop(self) -> None:
        dt = 1.0 / FPS
        while not self._stop.is_set():
            t0 = time.perf_counter()
            self._step(dt)
            elapsed = time.perf_counter() - t0
            time.sleep(max(0.0, dt - elapsed))

    def _step(self, dt: float) -> None:
        with self._lock:
            goto = self._goto
            joint_jog = dict(self._joint_jog)
            cart_jog = dict(self._cart_jog)
            q = self._q
        if goto is not None:
            self._goto_step()
        elif any(joint_jog.values()):
            self._joint_step(dt, q, joint_jog)
        elif any(cart_jog.values()):
            self._cart_step(dt, q, cart_jog)
        self._refresh_meas()

    def _goto_step(self) -> None:
        with self._lock:
            g = self._goto
            q = self._q
            if g is None:
                return
            i = g["i"] + 1
            a = min(i / g["n"], 1.0)
            kind = g["kind"]
            n = g["n"]
            q0 = g.get("q0")
            q1 = g.get("q1")
            xyz0 = g.get("xyz0")
            xyz1 = g.get("xyz1")
            R0 = g.get("R0")
            R1 = g.get("R1")
        if kind == "joint":
            q_next = (1.0 - a) * q0 + a * q1
        else:
            xyz = (1.0 - a) * xyz0 + a * xyz1
            R = R0 @ pin.exp3(a * pin.log3(R0.T @ R1))
            q_next = self._kin.servo_toward(q, xyz, R_ref=R, ori_weight=1.0)
        ok = self._commit(q_next, allow_lift_escape=False)
        with self._lock:
            if self._goto is None:
                return
            if not ok:
                self._goto = None
                if not self._fault:
                    self._fault = "go-to blocked"
            elif i >= n:
                self._goto = None
                self._fault = ""
            else:
                self._goto["i"] = i

    def _joint_step(self, dt: float, q: np.ndarray, joint_jog: dict[str, int]) -> None:
        dq = np.zeros_like(q)
        for name, sign in joint_jog.items():
            if sign == 0:
                continue
            if name == GRIPPER_JOINT:
                span = abs(GRIPPER_OPEN_CAD_DEG - GRIPPER_CLOSED_CAD_DEG)
                vel = (span / 100.0) * GRIPPER_VEL_UNIT_S
            else:
                vel = JOINT_VEL_DEG_S
            dq[self._kin.q_index(name)] = JOINT_SIGN[name] * sign * vel * DEG2RAD * dt
        self._commit(q + dq)

    def _cart_step(self, dt: float, q: np.ndarray, cart_jog: dict[str, int]) -> None:
        pose = self._kin.forward_tcp(q)
        translating = any(cart_jog[k] for k in ("x", "y", "z"))
        rotating = any(cart_jog[k] for k in ("wx", "wy", "wz"))
        with self._lock:
            jog_vel = float(self._jog_vel_mps)
            jog_rot = float(self._jog_rot_rad_s)
            meas_xyz = (
                None
                if self._mode != "real" or self._pose_meas is None
                else self._pose_meas.xyz_m.copy()
            )
            if self._target_xyz is None or self._target_rot is None:
                self._target_xyz = pose.xyz_m.copy()
                self._target_rot = pose.rotation.copy()
            # XYZ jog buttons are project base (Rz180°); IK targets stay URDF.
            v_user = np.array([cart_jog["x"], cart_jog["y"], cart_jog["z"]], dtype=float)
            self._target_xyz = self._target_xyz + jog_vel * dt * (R_URDF_FROM_USER @ v_user)
            # Cap vs the arm, not vs q_cmd. IK must keep accumulating on q_cmd
            # or each frame's joint goal stays < 1 Feetech tick and nothing moves.
            ref = meas_xyz if meas_xyz is not None else pose.xyz_m
            lead = self._target_xyz - ref
            cmd_user = np.array(
                [cart_jog["x"] != 0, cart_jog["y"] != 0, cart_jog["z"] != 0],
                dtype=bool,
            )
            # Which URDF axes are being driven (for lead cap).
            cmd = np.abs(R_URDF_FROM_USER) @ cmd_user.astype(float) > 0.5
            if translating and np.any(cmd):
                lead_cmd = np.where(cmd, lead, 0.0)
                nlead = float(np.linalg.norm(lead_cmd))
                if nlead > MAX_TARGET_LEAD_M:
                    self._target_xyz[cmd] = ref[cmd] + lead[cmd] * (MAX_TARGET_LEAD_M / nlead)
            if rotating:
                w_user = jog_rot * dt * np.array(
                    [cart_jog["wx"], cart_jog["wy"], cart_jog["wz"]], dtype=float
                )
                if self._rot_frame == ROT_FRAME_TCP:
                    # Body / TCP-local: Roll→TCP X, Pitch→TCP Y, Yaw→TCP Z.
                    self._target_rot = self._target_rot @ pin.exp3(w_user)
                else:
                    # Project base axes (Rz180° of URDF).
                    w = R_URDF_FROM_USER @ w_user
                    self._target_rot = pin.exp3(w) @ self._target_rot
            xyz = self._target_xyz.copy()
            rot = self._target_rot.copy()

        cmd_mask = (
            np.array([bool(cart_jog["x"]), bool(cart_jog["y"]), bool(cart_jog["z"])], dtype=bool)
            if translating
            else None
        )
        q_ik = self._kin.servo_toward(
            q,
            xyz,
            R_ref=rot if rotating else None,
            cmd_mask=cmd_mask,
            ori_weight=1.0 if rotating else 0.0,
        )
        # Spec: smooth the command. Raw IK nullspace (S1↔S4 on X) ticks the bus.
        q_next = (1.0 - CART_IK_BLEND) * q + CART_IK_BLEND * q_ik
        if not self._commit(q_next, rewind_cart=True):
            return
