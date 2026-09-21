"""Table pick-and-place sequence. No GUI, no YOLO.

Pick uses grasp TCP RPY. Place-down yaws so TCP Rx ∥ project-base +X.
Place Z = grasp Z.
State is just idle/running/done/fault plus a step index — not a BT.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from motion.controller import Controller
from motion.hw_controller import grip_100_to_user
from motion.robot_kinematics import (
    GRIPPER_JOINT,
    INIT_POSE_JOINTS_DEG,
    rotmat_to_rpy_deg,
    rpy_deg_to_rotmat,
)

PICK_LIFT_MM = 30.0
PLACE_APPROACH_MM = 50.0
# 픽앤플레이스 그리퍼 (0=닫힘, 100=완전 열림). 여닫힘 폭은 여기만 고친다.
GRIPPER_OPEN_100 = 30.0
GRIPPER_CLOSE_100 = 0.0
# 스텝과 스텝 사이 대기 (초). 0 이면 바로 다음.
STEP_PAUSE_S = 1.0
# Joint go-to reports this when already there — treat as success.
_ALREADY = "already at joint target"


@dataclass(frozen=True)
class PickPlaceStep:
    name: str
    kind: str  # ee | ee_joints | gripper | joints
    xyz_mm: np.ndarray | None = None
    rpy_deg: np.ndarray | None = None
    grip_100: float | None = None
    joints_deg: dict[str, float] | None = None


@dataclass
class PickPlaceRunner:
    ctrl: Controller
    lin_mps: float = 0.030
    rot_deg_s: float = 45.0
    _steps: list[PickPlaceStep] = field(default_factory=list)
    _i: int = 0
    _phase: str = "idle"  # idle | running | done | fault
    _interrupt: int = 0
    _dwell_until: float = 0.0
    status: str = ""

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def step_name(self) -> str:
        if not self._steps or self._i >= len(self._steps):
            return ""
        return self._steps[self._i].name

    def busy(self) -> bool:
        return self._phase == "running"

    def start(self, steps: list[PickPlaceStep]) -> bool:
        if not steps:
            self._phase = "fault"
            self.status = "빈 시퀀스"
            return False
        if self._phase == "running":
            self.status = "이미 실행 중"
            return False
        self._steps = list(steps)
        self._i = 0
        self._dwell_until = 0.0
        self._phase = "running"
        return self._launch()

    def abort(self, reason: str = "중단") -> None:
        self._phase = "fault"
        self.status = reason
        self.ctrl.stop_all()

    def tick(self) -> str:
        """Call from the GUI loop. Returns idle|running|done|fault."""
        if self._phase != "running":
            return self._phase
        if int(self.ctrl.interrupt_id()) != int(self._interrupt):
            self._phase = "fault"
            self.status = "중단 (Stop/조그)"
            return self._phase
        st = self.ctrl.snapshot()
        fault = str(st.fault or "")
        if fault and fault != _ALREADY:
            self._phase = "fault"
            self.status = fault
            return self._phase
        if self.ctrl.goto_active():
            self._dwell_until = 0.0
            self.status = self.step_name
            return self._phase
        if self._i + 1 >= len(self._steps):
            self._phase = "done"
            self.status = "완료"
            return self._phase
        pause = float(STEP_PAUSE_S)
        now = time.monotonic()
        if pause > 1e-9:
            if self._dwell_until <= 0.0:
                self._dwell_until = now + pause
                self.status = f"{self.step_name} 대기 {pause:.1f}s"
                return self._phase
            if now < self._dwell_until:
                self.status = f"{self.step_name} 대기 {pause:.1f}s"
                return self._phase
        self._dwell_until = 0.0
        self._i += 1
        if not self._launch():
            return self._phase
        return self._phase

    def _launch(self) -> bool:
        step = self._steps[self._i]
        self.status = step.name
        dur = 0.0
        if step.kind == "ee":
            dur = self.ctrl.start_ee_goto(
                np.asarray(step.xyz_mm, dtype=float),
                np.asarray(step.rpy_deg, dtype=float),
                lin_mps=float(self.lin_mps),
                rot_deg_s=float(self.rot_deg_s),
            )
        elif step.kind == "ee_joints":
            dur = self.ctrl.start_ee_goto_joints(
                np.asarray(step.xyz_mm, dtype=float),
                np.asarray(step.rpy_deg, dtype=float),
            )
        elif step.kind == "gripper":
            joints = dict(self.ctrl.snapshot().joints_deg)
            joints[GRIPPER_JOINT] = grip_100_to_user(float(step.grip_100))
            dur = self.ctrl.start_joint_goto(joints)
        elif step.kind == "joints":
            dur = self.ctrl.start_joint_goto(dict(step.joints_deg or INIT_POSE_JOINTS_DEG))
        else:
            self._phase = "fault"
            self.status = f"unknown step {step.kind}"
            return False
        self._interrupt = int(self.ctrl.interrupt_id())
        st = self.ctrl.snapshot()
        fault = str(st.fault or "")
        if self.ctrl.goto_active():
            return True
        if dur <= 0.0 and (not fault or fault == _ALREADY):
            return True
        if fault and fault != _ALREADY:
            self._phase = "fault"
            self.status = fault
            return False
        if dur <= 0.0:
            self._phase = "fault"
            self.status = "go-to 시작 실패"
            return False
        return True


def rpy_tcp_rx_parallel_base_x(rpy_deg: np.ndarray) -> np.ndarray:
    """Keep grasp pitch as far as possible; set TCP X exactly ∥ project-base X.

    Sign of TCP X (and 180° about X) is the one closer to ``rpy_deg``.
    """
    R0 = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    y0 = np.asarray(R0[:, 1], dtype=float)
    best_R = R0
    best_err = None
    for sx in (1.0, -1.0):
        x = np.array([sx, 0.0, 0.0], dtype=float)
        y = y0 - x * float(np.dot(y0, x))
        yn = float(np.linalg.norm(y))
        if yn < 1e-9:
            y = np.array([0.0, 0.0, 1.0], dtype=float)
            y = y - x * float(np.dot(y, x))
            yn = float(np.linalg.norm(y))
        y = y / yn
        z = np.cross(x, y)
        z = z / float(np.linalg.norm(z))
        y = np.cross(z, x)
        for R in (
            np.column_stack([x, y, z]),
            np.column_stack([x, -y, -z]),
        ):
            err = float(np.linalg.norm(R - R0, ord="fro"))
            if best_err is None or err < best_err:
                best_err = err
                best_R = R
    return rotmat_to_rpy_deg(best_R)


def build_pick_place_steps(
    *,
    p_xyz_mm: np.ndarray,
    p_rpy_deg: np.ndarray,
    g_xyz_mm: np.ndarray,
    g_rpy_deg: np.ndarray,
    drop_xy_mm: np.ndarray,
    gripper_close_100: float = GRIPPER_CLOSE_100,
    gripper_open_100: float = GRIPPER_OPEN_100,
    pick_lift_mm: float = PICK_LIFT_MM,
    place_approach_mm: float = PLACE_APPROACH_MM,
) -> list[PickPlaceStep]:
    """Linear table pick-and-place. Place Z from grasp TCP; place RPY aligns TCP Rx ∥ base X."""
    p_xyz = np.asarray(p_xyz_mm, dtype=float).reshape(3)
    g_xyz = np.asarray(g_xyz_mm, dtype=float).reshape(3)
    p_rpy = np.asarray(p_rpy_deg, dtype=float).reshape(3)
    g_rpy = np.asarray(g_rpy_deg, dtype=float).reshape(3)
    drop_rpy = rpy_tcp_rx_parallel_base_x(g_rpy)
    drop_xy = np.asarray(drop_xy_mm, dtype=float).reshape(2)
    z_g = float(g_xyz[2])
    z_lift = z_g + float(pick_lift_mm)
    z_hi = z_g + float(place_approach_mm)
    z_xy = max(z_lift, z_hi)
    close_100 = float(np.clip(gripper_close_100, 0.0, 100.0))
    open_100 = float(np.clip(gripper_open_100, 0.0, 100.0))

    def ee(name: str, xyz: np.ndarray, rpy: np.ndarray, *, joints: bool = False) -> PickPlaceStep:
        return PickPlaceStep(
            name=name,
            kind="ee_joints" if joints else "ee",
            xyz_mm=np.asarray(xyz, dtype=float).reshape(3),
            rpy_deg=np.asarray(rpy, dtype=float).reshape(3),
        )

    def grip(name: str, val: float) -> PickPlaceStep:
        return PickPlaceStep(name=name, kind="gripper", grip_100=float(val))

    gx, gy = float(g_xyz[0]), float(g_xyz[1])
    dx, dy = float(drop_xy[0]), float(drop_xy[1])
    steps = [
        ee("P", p_xyz, p_rpy),
        grip("그리퍼 열기", open_100),
        ee("G", g_xyz, g_rpy),
        grip("그리퍼 닫기", close_100),
        ee("집기 리프트", np.array([gx, gy, z_lift]), g_rpy),
        ee("드롭 XY", np.array([dx, dy, z_xy]), drop_rpy, joints=True),
    ]
    if z_xy > z_hi + 1e-6:
        steps.append(ee("드롭 위", np.array([dx, dy, z_hi]), drop_rpy))
    steps.extend(
        [
            ee("하강", np.array([dx, dy, z_g]), drop_rpy),
            grip("그리퍼 열기", open_100),
            ee("놓기 리프트", np.array([dx, dy, z_hi]), drop_rpy),
            PickPlaceStep(
                name="초기자세",
                kind="joints",
                joints_deg={k: float(v) for k, v in INIT_POSE_JOINTS_DEG.items()},
            ),
        ]
    )
    return steps
