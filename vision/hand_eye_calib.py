#!/usr/bin/env python3
"""SO-ARM eye-to-hand. 고정 카메라 + 끝단 ChArUco → T_base_cam (mm).

로봇 제어는 하지 않는다. 자세는 펜던트, 이 스크립트는 캡처와 Park만.
FK 프레임은 tcp. 보드를 끝단에 고정하고, YOLO와 같이 영상을 180° 회전한다.
공장 K의 주점(cx,cy)도 같이 뒤집어서 PnP에 쓴다.

키
  s / SPACE
           펜던트 TCP와 보드 영상을 저장. 코너 초록(16개+), Connect·토크 ON·정지.
  u        마지막 샘플 삭제.
  c        저장된 샘플로 Park를 풀어 eye_to_hand.json 에 쓴다 (12장 이상).
  q        종료. 12장 이상이면 c 와 같이 계산 후 저장.

  python vision/hand_eye_calib.py
  python vision/hand_eye_calib.py --compute-only
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from motion.pose_server import fetch_pose

from vision.calib import (
    CALIB_DIR,
    load_intrinsics,
    save_handeye,
    intrinsics_for_rotate180,
)
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180, grab_bgr, open_camera
from vision.charuco import (
    MARKER_MM,
    MIN_CORNERS,
    SQUARE_MM,
    SQUARES_X,
    SQUARES_Y,
    detect_board,
    draw_detection,
    make_board,
    match_points,
)
from vision.transforms import invert_T, rt_to_T

SAMPLES_DIR = CALIB_DIR / "handeye_tcp"
GRIPPER_FRAME = "tcp"
MIN_SOLVE = 12
RECOMMENDED = 20
XYZ_SPAN_MAX_MM = 2.0
RPY_SPAN_MAX_DEG = 1.5
PNP_RMS_MAX_PX = 0.8
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


def calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c):
    fn = getattr(cv2, "calibrateHandEye", None)
    if fn is not None:
        return fn(R_g2b, t_g2b, R_t2c, t_t2c, method=cv2.CALIB_HAND_EYE_PARK)
    n = len(R_g2b)
    Hg = [rt_to_T(R_g2b[i], t_g2b[i]) for i in range(n)]
    Hc = [rt_to_T(R_t2c[i], t_t2c[i]) for i in range(n)]
    M = np.zeros((3, 3))
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            Hgij = invert_T(Hg[j]) @ Hg[i]
            Hcij = Hc[j] @ invert_T(Hc[i])
            a, _ = cv2.Rodrigues(Hgij[:3, :3])
            b, _ = cv2.Rodrigues(Hcij[:3, :3])
            M += b.reshape(3, 1) @ a.reshape(1, 3)
            pairs.append((Hgij, Hcij))
    if len(pairs) < 2:
        raise RuntimeError("Hand-eye 실패 (회전 부족). 끝단을 30° 이상 돌려 다시 찍으세요.")
    w, V = np.linalg.eigh(M.T @ M)
    w, V = w[::-1], V[:, ::-1]
    R = V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T @ M.T
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    C = np.vstack([np.eye(3) - Hgij[:3, :3] for Hgij, _ in pairs])
    d = np.concatenate([Hgij[:3, 3] - R @ Hcij[:3, 3] for Hgij, Hcij in pairs])
    t = np.linalg.lstsq(C, d, rcond=None)[0]
    return R, t.reshape(3, 1)


def solve_eye_to_hand(T_base_tcp_list, T_cam_board_list) -> np.ndarray:
    if len(T_base_tcp_list) != len(T_cam_board_list):
        raise ValueError("tcp 포즈 수와 보드 포즈 수가 다릅니다.")
    if len(T_base_tcp_list) < 3:
        raise ValueError("eye-to-hand 샘플이 부족합니다 (3장 이상).")
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for T_bt, T_cb in zip(T_base_tcp_list, T_cam_board_list):
        T_tb = invert_T(np.asarray(T_bt, dtype=float))
        T_cb = np.asarray(T_cb, dtype=float)
        R_g2b.append(T_tb[:3, :3])
        t_g2b.append(T_tb[:3, 3])
        R_t2c.append(T_cb[:3, :3])
        t_t2c.append(T_cb[:3, 3])
    R, t = calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c)
    return rt_to_T(R, t)


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
    return payloads[mid], xyz_span.tolist(), rpy_span.tolist()


def list_samples(samples_dir: Path) -> list[Path]:
    return sorted(samples_dir.glob("*.png"))


def robust_pnp(obj, img, K, dist):
    obj = np.asarray(obj, dtype=np.float32).reshape(-1, 3)
    img = np.asarray(img, dtype=np.float32).reshape(-1, 2)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj,
        img,
        K,
        dist,
        iterationsCount=200,
        reprojectionError=1.5,
        confidence=0.999,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    min_inliers = max(12, int(np.ceil(len(obj) * 0.7)))
    if not ok or inliers is None or len(inliers) < min_inliers:
        return None
    use = np.asarray(inliers, dtype=int).reshape(-1)
    refine = getattr(cv2, "solvePnPRefineLM", None)
    if refine is not None:
        rvec, tvec = refine(obj[use], img[use], K, dist, rvec, tvec)
    projected, _ = cv2.projectPoints(obj[use], rvec, tvec, K, dist)
    error = projected.reshape(-1, 2) - img[use]
    rms = float(np.sqrt(np.mean(np.sum(error * error, axis=1))))
    return rvec, tvec, len(use), rms


def mean_rigid(transforms):
    result = np.eye(4)
    R_sum = np.sum([T[:3, :3] for T in transforms], axis=0)
    U, _, Vt = np.linalg.svd(R_sum)
    result[:3, :3] = U @ Vt
    if np.linalg.det(result[:3, :3]) < 0:
        U[:, -1] *= -1
        result[:3, :3] = U @ Vt
    result[:3, 3] = np.median(np.stack([T[:3, 3] for T in transforms]), axis=0)
    return result


def robust_limit(values, floor):
    med = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - med)))
    return max(floor, med + 3.0 * max(mad, 1e-6))


def compute(samples_dir: Path) -> np.ndarray:
    K_raw, dist_raw, k_path = load_intrinsics()
    K, dist = intrinsics_for_rotate180(K_raw, dist_raw, FRAME_WIDTH, FRAME_HEIGHT)
    print(
        f"PnP 180°  cx={K[0, 2]:.2f} cy={K[1, 2]:.2f}  "
        f"(공장 {K_raw[0, 2]:.2f},{K_raw[1, 2]:.2f})"
    )
    board = make_board()
    names = []
    T_cam_board_list = []
    T_base_tcp_list = []
    reprojection_rms = []

    for png in list_samples(samples_dir):
        meta_path = png.with_suffix(".json")
        if not meta_path.is_file():
            print("skip", png.name, "json 없음")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if not meta.get("rotate_180"):
            print("skip", png.name, "이전 샘플(회전 없음). 지우고 다시 찍으세요.")
            continue
        frame = cv2.imread(str(png))
        if frame is None:
            print("skip", png.name, "이미지 없음")
            continue
        det = detect_board(frame, board)
        if not det.ok:
            print("skip", png.name, "corners", det.n_corners)
            continue
        obj, img = match_points(det, board)
        if obj is None:
            print("skip", png.name, "match")
            continue
        solved = robust_pnp(obj, img, K, dist)
        if solved is None:
            print("skip", png.name, "RANSAC PnP")
            continue
        rvec, tvec, n_inlier, rms = solved
        if rms > PNP_RMS_MAX_PX:
            print("skip", png.name, f"reprojection RMS={rms:.3f}px")
            continue
        T_cam_board_list.append(rt_to_T(cv2.Rodrigues(rvec)[0], tvec))
        T_base_tcp_list.append(np.asarray(meta["T_base_tcp"], dtype=float))
        reprojection_rms.append(rms)
        names.append(png.name)
        print(f"ok {png.name} corners={det.n_corners} inliers={n_inlier} rms={rms:.3f}px")

    n_detected = len(names)
    if n_detected < MIN_SOLVE:
        raise RuntimeError(
            f"유효 샘플 {n_detected}장 — {RECOMMENDED}장 정도를 다시 찍으세요 (최소 {MIN_SOLVE})"
        )

    def calibrate_indices(indices):
        return solve_eye_to_hand(
            [T_base_tcp_list[i] for i in indices],
            [T_cam_board_list[i] for i in indices],
        )

    def tcp_board_residuals(T_base_cam, indices):
        transforms = [
            invert_T(T_base_tcp_list[i]) @ T_base_cam @ T_cam_board_list[i]
            for i in indices
        ]
        center = mean_rigid(transforms)
        t_error = np.array([np.linalg.norm(T[:3, 3] - center[:3, 3]) for T in transforms])
        r_error = np.array(
            [
                np.degrees(np.linalg.norm(cv2.Rodrigues(center[:3, :3].T @ T[:3, :3])[0]))
                for T in transforms
            ]
        )
        return transforms, center, t_error, r_error

    all_indices = list(range(n_detected))
    T_initial = calibrate_indices(all_indices)
    _, _, t_error0, r_error0 = tcp_board_residuals(T_initial, all_indices)
    t_limit = robust_limit(t_error0, 5.0)
    r_limit = robust_limit(r_error0, 2.0)
    keep = [
        i
        for i, (te, re) in enumerate(zip(t_error0, r_error0))
        if te <= t_limit and re <= r_limit
    ]
    rejected = [names[i] for i in all_indices if i not in keep]
    if len(keep) < MIN_SOLVE:
        raise RuntimeError(
            f"이상치 제거 후 {len(keep)}장. 보드 고정과 토크 ON 정지를 확인하고 다시 찍으세요."
        )

    T_base_cam = calibrate_indices(keep)
    _, T_tb_mean, t_error, r_error = tcp_board_residuals(T_base_cam, keep)

    loo_trans_error = []
    loo_rot_error = []
    for removed in keep:
        subset = [i for i in keep if i != removed]
        T_loo = calibrate_indices(subset)
        loo_trans_error.append(float(np.linalg.norm(T_loo[:3, 3] - T_base_cam[:3, 3])))
        dR = T_base_cam[:3, :3].T @ T_loo[:3, :3]
        loo_rot_error.append(float(np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0]))))

    rms = np.asarray(reprojection_rms, dtype=float)
    print(f"사용 {len(keep)}/{n_detected}장, 제거 {len(rejected)}장")
    if rejected:
        print("제거 샘플", rejected)
    print(
        "PnP RMS px mean/max",
        round(float(rms.mean()), 3),
        round(float(rms.max()), 3),
    )
    print("T_base_cam\n", T_base_cam)
    print("카메라 원점 in base mm", np.round(T_base_cam[:3, 3], 2))
    print(
        "tcp-보드 residual mm mean/max",
        round(float(t_error.mean()), 2),
        round(float(t_error.max()), 2),
    )
    print(
        "tcp-보드 residual deg mean/max",
        round(float(r_error.mean()), 3),
        round(float(r_error.max()), 3),
    )
    print("leave-one-out T 변화 mm max", round(float(np.max(loo_trans_error)), 2))
    print("leave-one-out R 변화 deg max", round(float(np.max(loo_rot_error)), 3))

    payload = {
        "frame": "T_base_cam",
        "gripper_frame": GRIPPER_FRAME,
        "unit": "mm",
        "method": "charuco_ransac_pnp_handeye_park_eye_to_hand",
        "intrinsics": str(k_path),
        "n_samples_detected": n_detected,
        "n_samples": len(keep),
        "rejected_samples": rejected,
        "squares": [SQUARES_X, SQUARES_Y],
        "square_mm": SQUARE_MM,
        "marker_mm": MARKER_MM,
        "min_corners": MIN_CORNERS,
        "rotate_180": True,
        "reprojection_rms_px_mean": float(rms.mean()),
        "reprojection_rms_px_max": float(rms.max()),
        "tcp_board_translation_residual_mm_mean": float(t_error.mean()),
        "tcp_board_translation_residual_mm_max": float(t_error.max()),
        "tcp_board_rotation_residual_deg_mean": float(r_error.mean()),
        "tcp_board_rotation_residual_deg_max": float(r_error.max()),
        "leave_one_out_translation_change_mm_max": float(np.max(loo_trans_error)),
        "leave_one_out_rotation_change_deg_max": float(np.max(loo_rot_error)),
        "T_base_cam": T_base_cam.tolist(),
        "T_tcp_board": T_tb_mean.tolist(),
        "t_base_cam_mm": T_base_cam[:3, 3].tolist(),
    }
    dest = save_handeye(payload)
    print("저장", dest)
    return T_base_cam


def capture_loop(samples_dir: Path) -> bool:
    board = make_board()
    samples_dir.mkdir(parents=True, exist_ok=True)
    saved = list_samples(samples_dir)
    cap = open_camera()
    print(f"저장 폴더 {samples_dir.name}, 이미 {len(saved)}장. s/SPACE/u/c/q")
    print("자세는 펜던트, 영상은 YOLO와 같이 180° 회전합니다.")
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
    compute_now = False
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
                    f"saved={len(saved)}  pendant OK  xyz={xyz}  s/SPACE u/c/q",
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
                compute_now = len(saved) >= MIN_SOLVE
                break
            if key == ord("c"):
                if len(saved) < MIN_SOLVE:
                    print(f"계산 안 함: {len(saved)}장 (최소 {MIN_SOLVE})")
                    continue
                compute_now = True
                break
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
                    mid, xyz_span, rpy_span = stable_tcp()
                except RuntimeError as exc:
                    print("저장 안 함:", exc)
                    continue
                stem = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                png = samples_dir / f"{stem}.png"
                cv2.imwrite(str(png), frame)
                xyz = np.asarray(mid["tcp_xyz_mm"], dtype=float)
                meta = {
                    "gripper_frame": GRIPPER_FRAME,
                    "T_base_tcp": mid["T_base_tcp"],
                    "tcp_xyz_mm": xyz.tolist(),
                    "tcp_rpy_deg": mid["tcp_rpy_deg"],
                    "joints_deg": {k: float(v) for k, v in mid["joints_deg"].items()},
                    "rotate_180": True,
                    "corners": det.n_corners,
                    "xyz_read_span_mm": xyz_span,
                    "rpy_read_span_deg": rpy_span,
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
                print(
                    f"저장 {png.name} corners={det.n_corners} "
                    f"xyz={np.round(xyz, 1).tolist()} total={len(saved)}"
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
    return compute_now


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-ARM eye-to-hand capture (tcp + ChArUco + Park)")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument(
        "--compute-only",
        action="store_true",
        help="카메라 없이 저장된 샘플만으로 계산",
    )
    return p.parse_args()


def main() -> None:
    _quiet_gtk()
    args = parse_args()
    samples_dir = args.samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    if args.compute_only:
        compute(samples_dir)
        return

    load_intrinsics()
    print(f"FK {GRIPPER_FRAME}  (포즈는 펜던트)")
    do_compute = capture_loop(samples_dir)

    if do_compute:
        compute(samples_dir)
    elif len(list_samples(samples_dir)) < MIN_SOLVE:
        print(f"계산 생략. {MIN_SOLVE}장 이상이면 q 또는 c, 또는 --compute-only")


if __name__ == "__main__":
    main()
