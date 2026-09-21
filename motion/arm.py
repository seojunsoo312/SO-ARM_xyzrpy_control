"""교육용 팔 API. 내부는 Controller. 단위는 펜던트와 같다 (mm, deg, 베이스 +X 앞)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from motion.controller import FPS, GOTO_MAX_S, GOTO_SETTLE_MAX_S, Controller
from motion.hw_controller import Hardware, grip_100_to_user
from motion.robot_kinematics import (
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


class ArmError(RuntimeError):
    """학생용 이동·연결 실패."""


class Arm:
    """가상/실기 팔. Meshcat이 같이 뜬다.

    arm = Arm()                       # mode="virtual"
    arm = Arm(mode="real")
    arm.move([30, 0, 0], "rel")
    arm.move([200, 50, 80], "abs")
    """

    def __init__(self, mode: str = "virtual", *, open_browser: bool = True) -> None:
        key = str(mode).strip().lower()
        if key not in _MODES:
            raise ArmError("mode는 'virtual' 또는 'real' 이어야 합니다")
        self._mode = key
        self._ctrl: Controller | None = None
        self._viz = None
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

    def where(self) -> dict[str, list[float]]:
        st = self._snapshot()
        self._show(st)
        return {
            "xyz_mm": [round(float(v), 2) for v in st.pose.xyz_mm],
            "rpy_deg": [round(float(v), 2) for v in st.pose.rpy_deg],
        }

    def move(self, xyz, mode: str, rpy=None) -> dict[str, list[float]]:
        how = str(mode).strip().lower()
        if how not in _MOVE_MODES:
            raise ArmError("move의 두 번째 인자는 'abs' 또는 'rel' 이어야 합니다")
        delta_or_abs = _vec3(xyz, "xyz는 mm 3개입니다")
        ctrl = self._require_ctrl()
        st = ctrl.snapshot()
        if how == "rel":
            target_xyz = np.asarray(st.pose.xyz_mm, dtype=float) + delta_or_abs
        else:
            target_xyz = delta_or_abs
        if rpy is None:
            target_rpy = np.asarray(st.pose.rpy_deg, dtype=float)
        else:
            target_rpy = _vec3(rpy, "rpy는 deg 3개입니다")
        dur = ctrl.start_ee_goto(target_xyz, target_rpy)
        self._wait_started(dur, joint=False)
        return self.where()

    def grip(self, value: float) -> None:
        ctrl = self._require_ctrl()
        joints = dict(ctrl.snapshot().joints_deg)
        joints[GRIPPER_JOINT] = grip_100_to_user(float(value))
        dur = ctrl.start_joint_goto(joints)
        self._wait_started(dur, joint=True)

    def home(self) -> dict[str, list[float]]:
        dur = self._require_ctrl().start_joint_goto(dict(HOME_JOINTS_DEG))
        self._wait_started(dur, joint=True)
        return self.where()

    def initial(self) -> dict[str, list[float]]:
        """펜던트 초기자세로 관절 이동."""
        dur = self._require_ctrl().start_joint_goto(dict(INIT_POSE_JOINTS_DEG))
        self._wait_started(dur, joint=True)
        return self.where()

    def stop(self) -> None:
        self._require_ctrl().stop_all()
        self._show()

    def close(self) -> None:
        ctrl = self._ctrl
        self._ctrl = None
        self._viz = None
        if ctrl is not None:
            ctrl.stop()

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


def _vec3(value, err: str) -> np.ndarray:
    try:
        arr = np.asarray(value, dtype=float).reshape(3)
    except (TypeError, ValueError):
        raise ArmError(err) from None
    if not np.all(np.isfinite(arr)):
        raise ArmError(err)
    return arr
