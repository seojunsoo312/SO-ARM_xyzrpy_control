"""카메라에 물체 등록을 요청하고, 그 6D로 교육용 픽앤플레이스 좌표를 만든다."""

from __future__ import annotations

import json
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from motion.arm import Arm, ArmError
from motion.pick_place import rpy_tcp_rx_parallel_base_x
from motion.robot_kinematics import RobotKinematics


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


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
    teach = _teach_grasp()
    place = _request_place(teach)
    mins_mm, maxs_mm = _cad_bounds_mm(teach)
    d_mm, a_mm = _approach_mm(teach)
    kin, q = _seed(arm)
    auto = teach.compute_auto_pre(
        place,
        mins_mm,
        maxs_mm,
        d_mm,
        a_mm,
        kin=kin,
        q=q,
    )
    p_xyz, p_rpy = teach.T_to_xyzrpy(auto.T_base)
    g_xyz, g_rpy = teach.T_to_xyzrpy(auto.T_grasp_base)
    place_rpy = rpy_tcp_rx_parallel_base_x(np.asarray(g_rpy, dtype=float))
    return PickTargets(
        p_xyz=_vec(p_xyz),
        p_rpy=_vec(p_rpy),
        g_xyz=_vec(g_xyz),
        g_rpy=_vec(g_rpy),
        place_rpy=_vec(place_rpy),
    )


def _teach_grasp():
    root = _project_root()
    pendant = root / "pendant"
    for path in (root, pendant):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    import teach_grasp

    return teach_grasp


def _request_place(teach):
    """등록 요청을 보내고, 이번 요청의 응답 6D를 돌려준다."""
    req_id = uuid.uuid4().hex
    path = teach.REGISTER_REQUEST_JSON
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"id": req_id}), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        raise ArmError(teach._REG_NO_REPLY) from exc
    data = _wait_register(teach, req_id)
    return _place_from_status(teach, data)


def _wait_register(teach, req_id: str) -> dict:
    start = time.monotonic()
    ack_deadline = start + float(teach.REG_ACK_S)
    job_deadline = start + float(teach.REG_JOB_S)
    accepted = False
    while True:
        now = time.monotonic()
        data = _read_status(teach)
        if isinstance(data, dict) and data.get("id") == req_id:
            state = str(data.get("state") or "")
            if state == "run":
                accepted = True
            elif state == "busy":
                raise ArmError("이미 등록 중입니다.")
            elif state == "no_box":
                raise ArmError(teach._REG_NO_BOX)
            elif state == "fail":
                raise ArmError(teach._REG_FAIL)
            elif state == "ok":
                return data
        if not accepted and now >= ack_deadline:
            raise ArmError(teach._REG_NO_REPLY)
        if accepted and now >= job_deadline:
            raise ArmError(teach._REG_NO_REPLY)
        time.sleep(float(teach.REG_POLL_MS) / 1000.0)


def _read_status(teach) -> dict | None:
    try:
        data = json.loads(teach.REGISTER_STATUS_JSON.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _place_from_status(teach, data: dict):
    xyz = data.get("xyz_mm")
    rpy = data.get("rpy_deg")
    if not isinstance(xyz, list) or not isinstance(rpy, list) or len(xyz) < 3 or len(rpy) < 3:
        raise ArmError(teach._REG_FAIL)
    try:
        return teach.PlacePose(
            xyz_mm=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            rpy_deg=(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        )
    except (TypeError, ValueError) as exc:
        raise ArmError(teach._REG_FAIL) from exc


def _cad_bounds_mm(teach) -> tuple[np.ndarray, np.ndarray]:
    try:
        verts_m, _faces = teach.load_cad_mesh_m(teach.cad_mesh_path())
    except Exception as exc:
        raise ArmError(f"CAD를 읽지 못했습니다 ({exc})") from exc
    verts_mm = np.asarray(verts_m, dtype=float).reshape(-1, 3) * 1000.0
    return verts_mm.min(axis=0), verts_mm.max(axis=0)


def _approach_mm(teach) -> tuple[float, float]:
    root = teach._parse_model_yaml(teach.CAD_YAML)
    d_mm = teach._scalar(root.get("approach_d_mm"), teach.DEFAULT_APPROACH_D_MM)
    a_mm = teach._scalar(root.get("approach_a_mm"), teach.DEFAULT_APPROACH_A_MM)
    return float(d_mm), float(a_mm)


def _seed(arm: Arm | None):
    if arm is not None:
        ctrl = arm._ctrl
        if ctrl is None:
            raise ArmError("이미 종료된 Arm 입니다")
        return ctrl._kin, ctrl.snapshot().q
    urdf = _project_root() / "pendant" / "SO101_6DOF.urdf"
    kin = RobotKinematics(urdf)
    return kin, kin.q_home()


def _vec(value) -> list[float]:
    arr = np.asarray(value, dtype=float).reshape(3)
    out: list[float] = []
    for v in arr:
        r = round(float(v), 2)
        out.append(0.0 if r == 0.0 else r)
    return out
