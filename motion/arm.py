"""교육용 팔 API. 내부는 Controller. 단위는 펜던트와 같다 (mm, deg, 베이스 +X 앞)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from motion.controller import (
    DEFAULT_SPEED_PCT,
    FPS,
    GOTO_MAX_S,
    GOTO_SETTLE_MAX_S,
    JOG_ROT_DEG_S,
    JOG_ROT_MAX_DEG_S,
    JOG_ROT_MIN_DEG_S,
    JOG_VEL_MAX_MPS,
    JOG_VEL_MIN_MPS,
    JOINT_VEL_DEG_S,
    JOINT_VEL_MIN_DEG_S,
    Controller,
    speed_pct_to_joint_vel,
)
from motion.hw_controller import Hardware, grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT,
    HOME_JOINTS_DEG,
    INIT_POSE_JOINTS_DEG,
    RobotKinematics,
)

_PROJECT = Path(__file__).resolve().parent.parent
_PENDANT = _PROJECT / "pendant"
_DEFAULT_URDF = _PENDANT / "SO101_6DOF.urdf"
_ALREADY = "already at joint target"
_MODES = ("virtual", "real")
_MOVE_MODES = ("abs", "rel")
_LIN_MM_S = 25.0
_LIN_MIN_MM_S = JOG_VEL_MIN_MPS * 1000.0
_LIN_MAX_MM_S = JOG_VEL_MAX_MPS * 1000.0
_JOINT_DEG_S = round(speed_pct_to_joint_vel(DEFAULT_SPEED_PCT))


class ArmError(RuntimeError):
    """학생용 이동·연결 실패."""


class Arm:
    """가상/실기 팔. Meshcat이 같이 뜬다.

    arm = Arm()                       # mode="virtual"
    arm = Arm(mode="real")
    arm.moveL([30, 0, 0], "rel")
    arm.moveJ([0, 10, 0, 0, 0, 0], "rel")
    """

    def __init__(self, mode: str = "virtual", *, open_browser: bool = True) -> None:
        key = str(mode).strip().lower()
        if key not in _MODES:
            raise ArmError("mode는 'virtual' 또는 'real' 이어야 합니다")
        self._mode = key
        self._ctrl: Controller | None = None
        self._viz = None
        self._owned = True
        self._watch_stop = False
        self._motion_epoch = 0
        kin = RobotKinematics(_DEFAULT_URDF)
        hardware = Hardware() if key == "real" else None
        try:
            self._ctrl = Controller(kin, hardware=hardware)
            self._ctrl.start(pose_server=False)
            if key == "real":
                self._ctrl.connect()
            self._viz = self._open_visualizer(kin, open_browser=open_browser)
            self._show()
            url = getattr(self._viz, "url", "") or ""
            print(f"Arm      mode={self._mode}")
            if url:
                print(f"Meshcat  {url}")
        except ArmError:
            self.close()
            raise
        except Exception as exc:
            self.close()
            if key == "real":
                raise ArmError(f"실기 연결 실패 ({exc})") from exc
            raise ArmError(str(exc)) from exc

    @classmethod
    def attach(cls, controller: Controller) -> Arm:
        """펜던트가 이미 돌리는 컨트롤러에 붙인다. Meshcat·시리얼은 새로 열지 않는다."""
        arm = cls.__new__(cls)
        arm._mode = "attached"
        arm._ctrl = controller
        arm._viz = None
        arm._owned = False
        arm._watch_stop = True
        arm._motion_epoch = controller.interrupt_id()
        return arm

    def where(self) -> dict[str, list[float] | float]:
        st = self._snapshot()
        self._show(st)
        joints = st.joints_deg
        return {
            "joints_deg": [_round1(joints[name]) for name in ARM_JOINT_NAMES],
            "gripper": _round1(grip_user_to_100(float(joints[GRIPPER_JOINT]))),
            "xyz_mm": [_round1(v) for v in st.pose.xyz_mm],
            "rpy_deg": [_round1(v) for v in st.pose.rpy_deg],
        }

    def moveL(
        self, xyz, mode: str, rpy=None, speed=None, rpy_speed=None, path: str = "line"
    ) -> dict[str, list[float] | float]:
        how = str(mode).strip().lower()
        if how not in _MOVE_MODES:
            raise ArmError("moveL의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다")
        route = str(path).strip().lower()
        if route not in ("line", "joint"):
            raise ArmError("path는 'line' 또는 'joint' 이어야 합니다")
        delta_or_abs = _vec3(xyz, "xyz는 mm 3개입니다")
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        st = ctrl.snapshot()
        if how == "rel":
            target_xyz = np.asarray(st.pose.xyz_mm, dtype=float) + delta_or_abs
        else:
            target_xyz = delta_or_abs
        if rpy is None:
            target_rpy = np.asarray(st.pose.rpy_deg, dtype=float)
        else:
            target_rpy = _vec3(rpy, "rpy는 deg 3개입니다")
        if route == "joint":
            if rpy_speed is not None:
                raise ArmError("path가 joint이면 rpy_speed는 쓰지 않습니다")
            joint_deg_s = _bounded(
                speed,
                _JOINT_DEG_S,
                JOINT_VEL_MIN_DEG_S,
                JOINT_VEL_DEG_S,
                "path가 joint이면 speed는 8–45 deg/s 입니다",
            )
            dur = ctrl.start_ee_goto_joints(
                target_xyz, target_rpy, speed_deg_s=joint_deg_s
            )
            self._mark_motion(ctrl)
            self._wait_started(dur, joint=True)
            return self.where()
        lin_mm_s = _bounded(
            speed, _LIN_MM_S, _LIN_MIN_MM_S, _LIN_MAX_MM_S, "speed는 5–45 mm/s 입니다"
        )
        rot_deg_s = _bounded(
            rpy_speed,
            JOG_ROT_DEG_S,
            JOG_ROT_MIN_DEG_S,
            JOG_ROT_MAX_DEG_S,
            "rpy_speed는 5–45 deg/s 입니다",
        )
        dur = ctrl.start_ee_goto(
            target_xyz,
            target_rpy,
            lin_mps=lin_mm_s / 1000.0,
            rot_deg_s=rot_deg_s,
        )
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=False)
        return self.where()

    def moveJ(self, joints, mode: str, gripper=None, speed=None) -> dict[str, list[float] | float]:
        how = str(mode).strip().lower()
        if how not in _MOVE_MODES:
            raise ArmError("moveJ의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다")
        delta_or_abs = _vec6(joints, "joints는 deg 6개입니다")
        joint_deg_s = _bounded(
            speed,
            _JOINT_DEG_S,
            JOINT_VEL_MIN_DEG_S,
            JOINT_VEL_DEG_S,
            "speed는 8–45 deg/s 입니다",
        )
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        current = ctrl.snapshot().joints_deg
        if how == "rel":
            target = {
                name: float(current[name]) + float(delta_or_abs[i])
                for i, name in enumerate(ARM_JOINT_NAMES)
            }
        else:
            target = {
                name: float(delta_or_abs[i]) for i, name in enumerate(ARM_JOINT_NAMES)
            }
        target[GRIPPER_JOINT] = _gripper_user(current[GRIPPER_JOINT], gripper, how)
        peak = max(abs(target[name] - float(current[name])) for name in ARM_JOINT_NAMES)
        duration_s = None if peak < 1e-3 else peak / joint_deg_s
        dur = ctrl.start_joint_goto(target, duration_s=duration_s)
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def grip(self, value: float) -> None:
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        joints = dict(ctrl.snapshot().joints_deg)
        joints[GRIPPER_JOINT] = grip_100_to_user(float(value))
        dur = ctrl.start_joint_goto(joints)
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)

    def home(self) -> dict[str, list[float] | float]:
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        dur = ctrl.start_joint_goto(dict(HOME_JOINTS_DEG))
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def initial(self) -> dict[str, list[float] | float]:
        """펜던트 초기자세로 관절 이동."""
        ctrl = self._require_ctrl()
        self._reject_external_stop(ctrl)
        dur = ctrl.start_joint_goto(dict(INIT_POSE_JOINTS_DEG))
        self._mark_motion(ctrl)
        self._wait_started(dur, joint=True)
        return self.where()

    def stop(self) -> None:
        self._require_ctrl().stop_all()
        self._show()

    def close(self) -> None:
        if not self._owned:
            self._ctrl = None
            self._viz = None
            return
        ctrl = self._ctrl
        viz = self._viz
        self._ctrl = None
        self._viz = None
        if viz is not None:
            try:
                viz.close()
            except Exception:
                pass
        if ctrl is not None:
            ctrl.stop()

    def _reject_external_stop(self, ctrl: Controller) -> None:
        if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
            raise ArmError("중단")

    def _mark_motion(self, ctrl: Controller) -> None:
        if self._watch_stop:
            self._motion_epoch = ctrl.interrupt_id()

    def _require_ctrl(self) -> Controller:
        if self._ctrl is None:
            raise ArmError("이미 종료된 Arm 입니다")
        return self._ctrl

    def _snapshot(self):
        return self._require_ctrl().snapshot()

    def _show(self, st=None) -> None:
        viz = self._viz
        if viz is None:
            return
        if st is None:
            if self._ctrl is None:
                return
            st = self._ctrl.snapshot()
        viz.display(st.q)

    def _wait_started(self, duration_s: float, *, joint: bool) -> None:
        ctrl = self._require_ctrl()
        st = ctrl.snapshot()
        fault = str(st.fault or "")
        if joint and fault == _ALREADY:
            self._show(st)
            return
        if not ctrl.goto_active():
            if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
                raise ArmError("중단")
            if fault:
                raise ArmError(fault)
            if duration_s <= 0.0:
                raise ArmError("동작 불가 (연결 중 또는 E-stop)")
            self._show(st)
            return
        deadline = time.monotonic() + GOTO_MAX_S + GOTO_SETTLE_MAX_S + max(float(duration_s), 0.0) + 2.0
        while True:
            st = ctrl.snapshot()
            self._show(st)
            fault = str(st.fault or "")
            if fault and not (joint and fault == _ALREADY):
                raise ArmError(fault)
            if self._watch_stop and ctrl.interrupt_id() != self._motion_epoch:
                raise ArmError("중단")
            if not ctrl.goto_active():
                return
            if time.monotonic() > deadline:
                ctrl.stop_all()
                raise ArmError("이동 시간 초과")
            time.sleep(1.0 / FPS)

    @staticmethod
    def _open_visualizer(kin: RobotKinematics, *, open_browser: bool):
        for path in (_PROJECT, _PENDANT):
            text = str(path)
            if text not in sys.path:
                sys.path.insert(0, text)
        from visualizer import Visualizer

        return Visualizer(kin, open_browser=open_browser)


def _round1(value) -> float:
    return round(float(value), 2)


def _vecn(value, n: int, err: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(n)
    except (TypeError, ValueError):
        raise ArmError(err) from None
    if not np.all(np.isfinite(arr)):
        raise ArmError(err)
    return arr


def _vec3(value, err: str) -> np.ndarray:
    return _vecn(value, 3, err)


def _vec6(value, err: str) -> np.ndarray:
    return _vecn(value, 6, err)


def _bounded(value, default: float, lo: float, hi: float, err: str) -> float:
    if value is None:
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArmError(err) from None
    if not np.isfinite(number) or number < lo or number > hi:
        raise ArmError(err)
    return number


def _gripper_user(current_user_deg: float, gripper, how: str) -> float:
    if gripper is None:
        return float(current_user_deg)
    try:
        asked = float(gripper)
    except (TypeError, ValueError):
        raise ArmError("gripper는 0–100 숫자입니다") from None
    if not np.isfinite(asked):
        raise ArmError("gripper는 0–100 숫자입니다")
    if how == "rel":
        asked = grip_user_to_100(float(current_user_deg)) + asked
    return grip_100_to_user(asked)
