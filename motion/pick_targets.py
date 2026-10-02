"""카메라에 물체 등록을 요청하고, 그 6D로 교육용 픽앤플레이스 좌표를 만든다."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cad.model import approach_mm, bounds_mm
from motion.arm import Arm, ArmError
from motion.grasp import PlacePose, T_to_xyzrpy, compute_auto_pre
from motion.pick_place import rpy_tcp_rx_parallel_base_x
from motion.robot_kinematics import DEFAULT_URDF, RobotKinematics
from yolo.pose.register_link import (
    REG_MSG_BUSY,
    REG_MSG_FAIL,
    REG_MSG_NO_BOX,
    REG_MSG_NO_REPLY,
    RegisterRequest,
)

# 등록이 끝난 state → 학생에게 보이는 오류.
_REG_ERRORS = {
    "busy": REG_MSG_BUSY,
    "no_box": REG_MSG_NO_BOX,
    "fail": REG_MSG_FAIL,
    "timeout": REG_MSG_NO_REPLY,
}


@dataclass(frozen=True)
class PickTargets:
    """대기 P, 집기 G, 놓기 자세. 위치는 mm, 자세는 deg."""

    p_xyz: list[float]
    p_rpy: list[float]
    g_xyz: list[float]
    g_rpy: list[float]
    place_rpy: list[float]


def load_pick_targets(arm: Arm | None = None) -> PickTargets:
    """카메라 창에 물체 등록을 요청하고, 그 6D로 대기·집기·놓기 자세를 만든다.

    펜던트 「물체 위치 불러오기」와 같은 요청이다. 결과가 올 때까지 이 줄에서 기다린다.
    arm을 주면 그 관절로 손목 자세를 고르고, 없으면 홈 관절을 쓴다.
    """
    place = _request_place()
    try:
        mins_mm, maxs_mm = bounds_mm()
    except Exception as exc:
        raise ArmError(f"CAD를 읽지 못했습니다 ({exc})") from exc
    d_mm, a_mm = approach_mm()
    kin, q = _seed(arm)
    auto = compute_auto_pre(
        place,
        mins_mm,
        maxs_mm,
        d_mm,
        a_mm,
        kin=kin,
        q=q,
    )
    p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
    g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
    place_rpy = rpy_tcp_rx_parallel_base_x(np.asarray(g_rpy, dtype=float))
    return PickTargets(
        p_xyz=_vec(p_xyz),
        p_rpy=_vec(p_rpy),
        g_xyz=_vec(g_xyz),
        g_rpy=_vec(g_rpy),
        place_rpy=_vec(place_rpy),
    )


def _request_place() -> PlacePose:
    """등록 요청을 보내고, 이번 요청의 응답 6D를 돌려준다."""
    try:
        req = RegisterRequest.send()
    except OSError as exc:
        raise ArmError(REG_MSG_NO_REPLY) from exc
    state = req.wait()
    if state != "ok":
        raise ArmError(_REG_ERRORS[state])
    return PlacePose(xyz_mm=req.xyz_mm, rpy_deg=req.rpy_deg)


def _seed(arm: Arm | None):
    if arm is not None:
        ctrl = arm._ctrl
        if ctrl is None:
            raise ArmError("이미 종료된 Arm 입니다")
        return ctrl._kin, ctrl.snapshot().q
    kin = RobotKinematics(DEFAULT_URDF)
    return kin, kin.q_home()


def _vec(value) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    out: list[float] = []
    for v in arr:
        r = round(float(v), 2)
        out.append(0.0 if r == 0.0 else r)
    return out
