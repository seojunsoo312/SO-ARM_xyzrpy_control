"""카메라 창(roi_cloud)에 물체 등록을 요청하고 결과를 받는다. JSON 파일 두 개로 주고받는다.

요청 쪽(펜던트 「물체 위치 불러오기」, load_pick_targets)이 요청 파일에 {"id"} 를 쓰면,
roi_cloud 가 같은 id 로 상태 파일에 state 를 쓴다.

  run     요청을 받아 등록 중
  ok      끝. xyz_mm, rpy_deg 가 같이 온다 (로봇 베이스, mm·deg)
  busy    이미 다른 등록 중이라 받지 않음
  no_box  바운딩 박스가 없음
  fail    등록 실패

요청 쪽은 REG_ACK_S 안에 run 이 없거나, run 뒤 REG_JOB_S 안에 끝나지 않으면 timeout 으로 본다.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

from yolo.config import RUNS_DIR

REGISTER_REQUEST_JSON = RUNS_DIR / "roi" / "register_request.json"
REGISTER_STATUS_JSON = RUNS_DIR / "roi" / "register_status.json"
REG_ACK_S = 2.0  # 요청 → run 까지
REG_JOB_S = 30.0  # run → 끝까지
REG_POLL_MS = 200
REG_MSG_RUN = "현재 등록 중입니다."
REG_MSG_BUSY = "이미 등록 중입니다."
REG_MSG_NO_BOX = "바운딩 박스가 없습니다. 카메라 창을 확인해 주세요."
REG_MSG_FAIL = "등록에 실패했습니다. 카메라 창을 확인해 주세요."
REG_MSG_NO_REPLY = "카메라 창이 응답하지 않습니다. 실행중인지 확인해주세요"
# poll() 이 이 중 하나를 돌려주면 끝. 나머지는 "wait"(아직 응답 없음), "run".
DONE_STATES = ("ok", "busy", "no_box", "fail", "timeout")


def _write_json(path: Path, payload: dict) -> None:
    """tmp 에 쓰고 바꿔 끼운다. 읽는 쪽이 반쯤 쓴 파일을 보지 않게."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None


# --- roi_cloud 쪽 ---


def read_request() -> str | None:
    """요청 id. 없거나 깨졌으면 None."""
    data = _read_json(REGISTER_REQUEST_JSON)
    rid = data.get("id") if data is not None else None
    if not isinstance(rid, str) or not rid:
        return None
    return rid


def write_status(req_id: str, state: str, *, xyz=None, rpy=None) -> None:
    """state: run, busy, no_box, fail, ok. ok 이면 xyz(mm), rpy(deg) 를 같이 준다."""
    payload: dict = {"id": str(req_id), "state": state}
    if xyz is not None and rpy is not None:
        payload["xyz_mm"] = [float(v) for v in xyz]
        payload["rpy_deg"] = [float(v) for v in rpy]
    _write_json(REGISTER_STATUS_JSON, payload)


# --- 요청 쪽 ---


class RegisterRequest:
    """등록 요청 하나. send() 로 보내고 poll() 이나 wait() 로 결과를 본다.

    GUI 는 타이머에서 poll() 을, 스크립트는 wait() 을 부른다.
    ok 이면 xyz_mm, rpy_deg 에 결과가 있다.
    """

    def __init__(self) -> None:
        self.id = uuid.uuid4().hex
        self.state = "wait"
        self.xyz_mm: tuple[float, float, float] | None = None
        self.rpy_deg: tuple[float, float, float] | None = None
        self._accepted = False
        self._ack_deadline = 0.0
        self._job_deadline = 0.0

    @classmethod
    def send(cls) -> RegisterRequest:
        """요청 파일을 쓴다. 쓰지 못하면 OSError."""
        req = cls()
        now = time.monotonic()
        req._ack_deadline = now + REG_ACK_S
        req._job_deadline = now + REG_JOB_S
        _write_json(REGISTER_REQUEST_JSON, {"id": req.id})
        return req

    @property
    def done(self) -> bool:
        return self.state in DONE_STATES

    def poll(self) -> str:
        """상태 파일을 한 번 읽고 state 를 돌려준다. 끝난 뒤에는 그 값을 그대로."""
        if self.done:
            return self.state
        data = _read_json(REGISTER_STATUS_JSON)
        if data is not None and data.get("id") == self.id:
            state = str(data.get("state") or "")
            if state == "run":
                self._accepted = True
                self.state = "run"
            elif state in ("busy", "no_box", "fail"):
                self.state = state
                return self.state
            elif state == "ok":
                self.state = "ok" if self._take_pose(data) else "fail"
                return self.state
        now = time.monotonic()
        deadline = self._job_deadline if self._accepted else self._ack_deadline
        if now >= deadline:
            self.state = "timeout"
        return self.state

    def wait(self) -> str:
        """끝날 때까지 이 줄에서 기다린다."""
        while not self.done:
            self.poll()
            if not self.done:
                time.sleep(REG_POLL_MS / 1000.0)
        return self.state

    def _take_pose(self, data: dict) -> bool:
        xyz = data.get("xyz_mm")
        rpy = data.get("rpy_deg")
        if not isinstance(xyz, list) or not isinstance(rpy, list) or len(xyz) < 3 or len(rpy) < 3:
            return False
        try:
            self.xyz_mm = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            self.rpy_deg = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
        except (TypeError, ValueError):
            return False
        return True
