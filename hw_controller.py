"""LeRobot SOFollower wrapper. IK/GUI-unaware. Bus 0° = calib Enter."""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from robot_kinematics import (
    ARM_JOINT_NAMES,
    GRIPPER_JOINT,
    LEROBOT_FROM_URDF,
    URDF_JOINT_NAMES,
    RobotKinematics,
)

STS3215_RESOLUTION = 4096
GRIPPER_MID = 50.0
DEFAULT_PORT = "/dev/so101_follower"
DEFAULT_ROBOT_ID = "follower"


def grip_user_to_100(user_deg: float) -> float:
    """Pendant S7 user-deg (HOME=0) → LeRobot gripper 0–100 (HOME=50)."""
    return float(np.clip(user_deg + GRIPPER_MID, 0.0, 100.0))


def grip_100_to_user(grip_100: float) -> float:
    """LeRobot gripper 0–100 → pendant S7 user-deg."""
    return float(np.clip(grip_100, 0.0, 100.0)) - GRIPPER_MID


def _enter_offsets_deg(calibration: dict) -> dict[str, float]:
    """LeRobot DEGREES zeros at range midpoint; shift so calib Enter (2047) is 0°."""
    half_turn = (STS3215_RESOLUTION - 1) // 2
    max_res = STS3215_RESOLUTION - 1
    offsets: dict[str, float] = {}
    for urdf_name in ARM_JOINT_NAMES:
        motor = LEROBOT_FROM_URDF[urdf_name]
        cal = calibration.get(motor)
        if cal is None:
            offsets[urdf_name] = 0.0
            continue
        mid = (float(cal.range_min) + float(cal.range_max)) / 2.0
        offsets[urdf_name] = (mid - half_turn) * 360.0 / max_res
    return offsets


class Hardware:
    """connect() reads encoders, syncs q, then enables torque. Never dumps a virtual pose."""

    def __init__(
        self,
        *,
        port: str = DEFAULT_PORT,
        robot_id: str = DEFAULT_ROBOT_ID,
        calibration_dir: Path | None = None,
        dof_mode: int = 7,
    ) -> None:
        self.port = port
        self.robot_id = robot_id
        self.calibration_dir = Path(calibration_dir) if calibration_dir is not None else None
        self.dof_mode = int(dof_mode)
        self._lock = threading.Lock()
        self._robot = None
        self._enter_off: dict[str, float] = {}
        self._torque = False

    @property
    def is_connected(self) -> bool:
        robot = self._robot
        return bool(robot is not None and robot.is_connected)

    @property
    def torque_enabled(self) -> bool:
        return bool(self.is_connected and self._torque)

    def connect(self, kin: RobotKinematics) -> np.ndarray:
        """Bus on, torque off, q from encoders, write that goal, then torque on. Returns Pinocchio q."""
        from lerobot.robots.so_follower import SO101FollowerConfig, SOFollower

        self.disconnect()
        kwargs: dict = {
            "port": self.port,
            "id": self.robot_id,
            "use_degrees": True,
            "dof_mode": self.dof_mode,
            "max_relative_target": None,
            "cameras": {},
        }
        if self.calibration_dir is not None:
            kwargs["calibration_dir"] = self.calibration_dir
        robot = SOFollower(SO101FollowerConfig(**kwargs))
        if not robot.calibration:
            raise RuntimeError(
                f"No calibration at {robot.calibration_fpath}. "
                "Use an existing follower.json (do not re-calibrate here)."
            )
        missing = [
            LEROBOT_FROM_URDF[n] for n in URDF_JOINT_NAMES if LEROBOT_FROM_URDF[n] not in robot.calibration
        ]
        if missing:
            raise RuntimeError(f"Calibration missing motors {missing}")

        try:
            robot.connect(calibrate=False)
        except Exception:
            try:
                robot.disconnect()
            except Exception:
                pass
            raise

        try:
            # configure() re-enables torque on exit; hold pose only after q_cmd = q_meas.
            with self._lock:
                robot.bus.disable_torque()
            self._enter_off = _enter_offsets_deg(robot.calibration)
            with self._lock:
                obs = robot.get_observation()
            q = kin.q_from_deg(self._bus_to_user(obs))
            action = self._user_to_bus(kin.joints_deg(q))
            with self._lock:
                robot.send_action(action)
                robot.bus.enable_torque()
            self._robot = robot
            self._torque = True
            return q
        except Exception:
            self._robot = None
            self._torque = False
            self._enter_off = {}
            try:
                robot.bus.disable_torque()
            except Exception:
                pass
            try:
                robot.disconnect()
            except Exception:
                pass
            raise

    def disconnect(self) -> None:
        robot = self._robot
        self._robot = None
        self._torque = False
        self._enter_off = {}
        if robot is None:
            return
        try:
            robot.bus.disable_torque()
        except Exception:
            pass
        try:
            robot.disconnect()
        except Exception:
            pass

    def disable_torque(self) -> None:
        self._torque = False
        robot = self._robot
        if robot is None:
            return
        with self._lock:
            robot.bus.disable_torque()

    def enable_torque(self) -> None:
        robot = self._robot
        if robot is None:
            return
        with self._lock:
            robot.bus.enable_torque()
        self._torque = True

    def read_q(self, kin: RobotKinematics) -> np.ndarray:
        robot = self._robot
        if robot is None:
            raise RuntimeError("hardware not connected")
        with self._lock:
            obs = robot.get_observation()
        user = self._bus_to_user(obs)
        return kin.q_from_deg(user)

    def write_q(self, kin: RobotKinematics, q: np.ndarray) -> None:
        robot = self._robot
        if robot is None:
            raise RuntimeError("hardware not connected")
        action = self._user_to_bus(kin.joints_deg(q))
        with self._lock:
            robot.send_action(action)

    def _bus_to_user(self, obs: dict) -> dict[str, float]:
        user: dict[str, float] = {}
        for urdf_name in ARM_JOINT_NAMES:
            motor = LEROBOT_FROM_URDF[urdf_name]
            bus = float(obs[f"{motor}.pos"])
            user[urdf_name] = bus + self._enter_off.get(urdf_name, 0.0)
        grip = float(obs.get("gripper.pos", GRIPPER_MID))
        user[GRIPPER_JOINT] = grip_100_to_user(grip)
        return user

    def _user_to_bus(self, user: dict[str, float]) -> dict[str, float]:
        action: dict[str, float] = {}
        for urdf_name in ARM_JOINT_NAMES:
            motor = LEROBOT_FROM_URDF[urdf_name]
            action[f"{motor}.pos"] = float(user[urdf_name]) - self._enter_off.get(urdf_name, 0.0)
        action["gripper.pos"] = grip_user_to_100(float(user[GRIPPER_JOINT]))
        return action
