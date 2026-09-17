#!/usr/bin/env python3
"""SO-ARM eye-to-hand 캡처. 고정 카메라 + 끝단 ChArUco → png+json.

로봇 제어는 하지 않는다. 자세는 펜던트, 이 스크립트는 저장만.
계산(T_base_cam)은 `handeye/compute.py`.
FK 프레임은 tcp. 보드를 끝단에 고정하고, YOLO와 같이 영상을 180° 회전한다.

베이스는 프로젝트 프레임(+X 전진). 펜던트 pose 서버가 그 기준으로
T_base_tcp 을 주므로, 저장되는 메타도 같은 베이스다.

키
  s / SPACE
           펜던트 TCP와 보드 영상을 저장. 코너 초록(16개+), Connect·토크 ON·정지.
  u        마지막 샘플 삭제.
  q        종료. 계산하지 않음.

찍을 때: 보드를 끝단에 단단히. 위치만 바꾸지 말고 rx/ry/rz 각 축 30° 이상.
같은 자세 반복보다 작업공간 구석·가운데를 섞는다.

  python vision/handeye/capture.py
  python vision/handeye/compute.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion.base_frame import USER_BASE_FRAME
from motion.pose_server import fetch_pose

from vision.calib import load_intrinsics
from vision.camera import ROTATE_180, grab_bgr, open_camera
from vision.handeye.charuco import (
    MARKER_MM,
    MIN_CORNERS,
    SQUARE_MM,
    SQUARES_X,
    SQUARES_Y,
    detect_board,
    draw_detection,
    make_board,
)
from vision.handeye.compute import (
    GRIPPER_FRAME,
    MIN_SOLVE,
    RECOMMENDED,
    SAMPLES_DIR,
    list_samples,
)

XYZ_SPAN_MAX_MM = 2.0
RPY_SPAN_MAX_DEG = 1.5
WIN = "hand-eye"
KR_FONT = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
PENDANT_OFF_LINES = (
    "펜던트를 먼저 켜세요.",
    "이 창을 끄고 다시 실행하세요.",
)


def _quiet_gtk() -> None:
    for key in ("GTK_MODULES", "GTK3_MODULES"):
        raw = os.environ.get(key)
        if not raw:
            continue
        os.environ[key] = ":".join(
            p for p in raw.split(":") if p and "canberra" not in p.lower()
        )


_KR_FONT_CACHE: dict[int, object] = {}


def _kr_font(size: int):
    from PIL import ImageFont

    cached = _KR_FONT_CACHE.get(size)
    if cached is not None:
        return cached
    font = None
    if KR_FONT.is_file():
        for index in (1, 0):
            try:
                font = ImageFont.truetype(str(KR_FONT), size=size, index=index)
                break
            except OSError:
                continue
    if font is None:
        font = ImageFont.load_default()
    _KR_FONT_CACHE[size] = font
    return font


def put_kr_lines(bgr: np.ndarray, lines: list[str], *, origin=(16, 16), size=32) -> np.ndarray:
    """Hangul overlay. Hershey putText cannot draw Korean."""
    from PIL import Image, ImageDraw

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(rgb)
    draw = ImageDraw.Draw(im)
    font = _kr_font(size)
    x, y = origin
    for line in lines:
        draw.text(
            (x, y),
            line,
            font=font,
            fill=(255, 40, 40),
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
        y += size + 10
    return cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)


def pendant_block_reason(payload: dict | None) -> str | None:
    if payload is None:
        return "off"
    if not payload.get("connected") or payload.get("mode") != "real":
        return "connect"
    if not payload.get("torque"):
        return "torque"
    return None


def overlay_for_reason(reason: str | None) -> list[str]:
    if reason == "off":
        return list(PENDANT_OFF_LINES)
    if reason == "connect":
        return ["펜던트에서 Connect 하세요."]
    if reason == "torque":
        return ["펜던트 토크가 꺼져 있습니다."]
    return []


def stable_tcp(samples: int = 5, settle_s: float = 0.5):
    reason = pendant_block_reason(fetch_pose())
    if reason is not None:
        lines = overlay_for_reason(reason)
        raise RuntimeError(" ".join(lines))
    time.sleep(settle_s)
    xyz, rpy, payloads = [], [], []
    for _ in range(samples):
        payload = fetch_pose()
        reason = pendant_block_reason(payload)
        if reason is not None or payload is None:
            lines = overlay_for_reason(reason)
            raise RuntimeError(" ".join(lines) if lines else "펜던트 포즈를 읽지 못했습니다.")
        xyz.append(np.asarray(payload["tcp_xyz_mm"], dtype=float))
        rpy.append(np.asarray(payload["tcp_rpy_deg"], dtype=float))
        payloads.append(payload)
        time.sleep(0.08)
    xyz = np.asarray(xyz, dtype=float)
    rpy = np.asarray(rpy, dtype=float)
    xyz_span = np.ptp(xyz, axis=0)
    rpy_span = np.ptp(rpy, axis=0)
    if np.any(xyz_span > XYZ_SPAN_MAX_MM) or np.any(rpy_span > RPY_SPAN_MAX_DEG):
        raise RuntimeError(
            f"TCP가 불안정합니다. xyz span={np.round(xyz_span, 2)} mm, "
            f"rpy span={np.round(rpy_span, 2)} deg"
        )
    mid = samples // 2
    err_rows = [
        np.asarray(p["err_xyz_mm"], dtype=float).reshape(3)
        for p in payloads
        if p.get("err_xyz_mm") is not None
    ]
    if err_rows:
        err_xyz = np.mean(np.stack(err_rows, axis=0), axis=0)
        ee_err = float(np.linalg.norm(err_xyz))
        err_xyz_mm = err_xyz.tolist()
    else:
        ee_err = None
        err_xyz_mm = None
    return payloads[mid], xyz_span.tolist(), rpy_span.tolist(), ee_err, err_xyz_mm


def capture_loop(samples_dir: Path) -> None:
    board = make_board()
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = list_samples(samples_dir)
    cap = open_camera()
    print(f"저장 폴더 {samples_dir.name}, 이미 {len(saved)}장. s/SPACE/u/q")
    print("자세는 펜던트, 영상은 YOLO와 같이 180° 회전합니다.")
    print("정확도: 보드 단단히, 정지 후 s. rx/ry/rz 각 30°+, 작업공간 여러 곳.")
    print("계산은 이 창이 아니라 python vision/handeye/compute.py")
    n_old = 0
    for png in saved:
        meta_path = png.with_suffix(".json")
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("rotate_180"):
            n_old += 1
    if n_old:
        print(
            f"경고: 회전 없는 예전 샘플 {n_old}장. "
            "계산에서 빠집니다. 폴더를 비우거나 u로 지우고 다시 찍으세요."
        )
    if fetch_pose() is None:
        print("펜던트가 없습니다. 펜던트를 먼저 켜고 이 창을 다시 실행하세요.")
    payload = fetch_pose()
    last_pose_t = time.monotonic()
    try:
        while True:
            frame = grab_bgr(cap, rotate_180=ROTATE_180)
            if frame is None:
                raise RuntimeError("컬러 프레임이 없습니다. Viewer를 끄세요.")
            det = detect_board(frame, board)
            vis = draw_detection(frame, det)
            now = time.monotonic()
            if now - last_pose_t >= 0.15:
                payload = fetch_pose()
                last_pose_t = now
            reason = pendant_block_reason(payload)
            if reason is not None:
                vis = put_kr_lines(vis, overlay_for_reason(reason))
                cv2.putText(
                    vis,
                    f"saved={len(saved)}",
                    (16, vis.shape[0] - 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            else:
                xyz = np.round(np.asarray(payload["tcp_xyz_mm"], dtype=float), 1).tolist()
                cv2.putText(
                    vis,
                    f"saved={len(saved)}  pendant OK  xyz={xyz}  s/SPACE u/q",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if det.ok else (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                break
            if key == ord("c"):
                print(
                    f"계산은 이 창이 아닙니다. {len(saved)}장 → "
                    "python vision/handeye/compute.py"
                )
                continue
            if key == ord("u") and saved:
                png = saved.pop()
                png.unlink(missing_ok=True)
                png.with_suffix(".json").unlink(missing_ok=True)
                print("삭제", png.name, "남은", len(saved))
            if key in (ord(" "), ord("s")):
                if reason is not None:
                    print("저장 안 함:", " ".join(overlay_for_reason(reason)))
                    continue
                if not det.ok:
                    print(f"저장 안 함: 코너 {det.n_corners}개, 최소 {MIN_CORNERS}개")
                    continue
                try:
                    mid, xyz_span, rpy_span, ee_err, err_xyz = stable_tcp()
                except RuntimeError as exc:
                    print("저장 안 함:", exc)
                    continue
                stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                png = samples_dir / f"{stem}.png"
                cv2.imwrite(str(png), frame)
                xyz = np.asarray(mid["tcp_xyz_mm"], dtype=float)
                meta = {
                    "gripper_frame": GRIPPER_FRAME,
                    "base_frame": USER_BASE_FRAME,
                    "T_base_tcp": mid["T_base_tcp"],
                    "tcp_xyz_mm": xyz.tolist(),
                    "tcp_rpy_deg": mid["tcp_rpy_deg"],
                    "joints_deg": {k: float(v) for k, v in mid["joints_deg"].items()},
                    "rotate_180": True,
                    "corners": det.n_corners,
                    "xyz_read_span_mm": xyz_span,
                    "rpy_read_span_deg": rpy_span,
                    # Pendant err: cmd−meas TCP, project base (same as GUI Δx/Δy/Δz).
                    "ee_err_mm": ee_err,
                    "err_xyz_mm": err_xyz,
                    "board": {
                        "squares": [SQUARES_X, SQUARES_Y],
                        "square_mm": SQUARE_MM,
                        "marker_mm": MARKER_MM,
                    },
                }
                png.with_suffix(".json").write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
                saved.append(png)
                if err_xyz is not None and ee_err is not None:
                    err_txt = (
                        f" |Δ|={ee_err:.2f} "
                        f"Δ=[{err_xyz[0]:+.2f},{err_xyz[1]:+.2f},{err_xyz[2]:+.2f}]"
                    )
                else:
                    err_txt = " err=n/a"
                print(
                    f"저장 {png.name} corners={det.n_corners} "
                    f"xyz={np.round(xyz, 1).tolist()}{err_txt} total={len(saved)}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()

    poses = []
    for png in list_samples(samples_dir):
        meta = json.loads(png.with_suffix(".json").read_text(encoding="utf-8"))
        poses.append(meta["tcp_rpy_deg"])
    print("총", len(saved), "장")
    if len(poses) >= 2:
        rpy = np.asarray(poses, dtype=float)
        rpy_span = np.ptp(rpy, axis=0)
        print("rx,ry,rz 범위(deg)", np.round(rpy_span, 1).tolist())
        if np.any(rpy_span < 30):
            print("회전이 작습니다. 각 축으로 30° 이상 변화를 주세요.")
    if len(saved) < RECOMMENDED:
        print(f"{len(saved)}장 — {RECOMMENDED}장 전후를 권장합니다.")
    if len(saved) >= MIN_SOLVE:
        print("다음: python vision/handeye/compute.py")
    else:
        print(f"계산하려면 {MIN_SOLVE}장 이상. 지금은 {len(saved)}장.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-ARM eye-to-hand capture (tcp + ChArUco)")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    return p.parse_args()


def main() -> None:
    _quiet_gtk()
    args = parse_args()
    samples_dir = args.samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)
    load_intrinsics()
    print(f"FK {GRIPPER_FRAME}  (포즈는 펜던트)")
    capture_loop(samples_dir)


if __name__ == "__main__":
    main()
