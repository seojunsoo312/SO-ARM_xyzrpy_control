#!/usr/bin/env python3
"""책상 위 ChArUco: 코너마다 PnP Z vs D2C 뎁스 Z (mm).

보드를 작업 영역 책상에 평평히 두고, Viewer·펜던트는 끈다.

  python vision/handeye/compare_pnp_depth.py

SPACE 한 프레임을 고정해서 숫자 출력. q 종료.
손-눈과 같이 640×480, 180° 회전, 공장 K·왜곡, SDK D2C.
PnP 코너를 T_base_cam 으로 올려 z_base 도 찍는다.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _quiet_gtk() -> None:
    for key in ("GTK_MODULES", "GTK3_MODULES"):
        raw = os.environ.get(key)
        if not raw:
            continue
        os.environ[key] = ":".join(
            p for p in raw.split(":") if p and "canberra" not in p.lower()
        )


_quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.calib import load_T_base_cam, load_intrinsics, intrinsics_for_rotate180  # noqa: E402
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180, rotate_rgbd  # noqa: E402
from vision.handeye.charuco import MIN_CORNERS, detect_board, draw_detection, make_board, match_points  # noqa: E402
from vision.handeye.compute import robust_pnp  # noqa: E402
from vision.rgbd import OrbbecV1  # noqa: E402
from vision.transforms import apply_T  # noqa: E402

Z_MIN_MM = 80.0
Z_MAX_MM = 1800.0

WIN = "pnp vs depth"
PATCH = 1  # 3×3 = radius 1


def _colorize_depth(depth_mm: np.ndarray) -> np.ndarray:
    vis = np.zeros((*depth_mm.shape[:2], 3), dtype=np.uint8)
    valid = (depth_mm > Z_MIN_MM) & (depth_mm < Z_MAX_MM)
    if not np.any(valid):
        return vis
    values = depth_mm[valid]
    center = float(np.median(values))
    low = max(Z_MIN_MM, center - 40.0)
    high = min(Z_MAX_MM, center + 40.0)
    norm = np.clip(depth_mm, low, high)
    norm = ((norm - low) / max(high - low, 1.0) * 255.0).astype(np.uint8)
    vis[valid] = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)[valid]
    return vis


def _depth_median(depth: np.ndarray, u: float, v: float, radius: int = PATCH):
    h, w = depth.shape[:2]
    x = int(round(u))
    y = int(round(v))
    if not (0 <= x < w and 0 <= y < h):
        return None
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float64)
    valid = patch[(patch > Z_MIN_MM) & (patch < Z_MAX_MM)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def compare_frame(bgr, depth, board, K, dist, T_base_cam: np.ndarray):
    det = detect_board(bgr, board)
    vis = draw_detection(bgr, det)
    if depth.shape[:2] != bgr.shape[:2]:
        return vis, None, (
            f"D2C 실패: color {bgr.shape[1]}x{bgr.shape[0]}  "
            f"depth {depth.shape[1]}x{depth.shape[0]}"
        )
    if not det.ok:
        return vis, None, f"코너 {det.n_corners}개 (최소 {MIN_CORNERS})"

    obj, img = match_points(det, board)
    if obj is None:
        return vis, None, "match 실패"

    solved = robust_pnp(obj, img, K, dist)
    if solved is None:
        return vis, None, "PnP 실패"
    rvec, tvec, n_in, rms = solved
    R, _ = cv2.Rodrigues(rvec)
    t = np.asarray(tvec, dtype=np.float64).reshape(3)
    obj_xyz = np.asarray(obj, dtype=np.float64).reshape(-1, 3)
    img_uv = np.asarray(img, dtype=np.float64).reshape(-1, 2)
    p_cam = (R @ obj_xyz.T).T + t
    p_base = apply_T(p_cam.astype(np.float32), T_base_cam)

    rows = []
    for (u, v), z_pnp, pb in zip(img_uv, p_cam[:, 2], p_base):
        z_d = _depth_median(depth, u, v)
        rows.append(
            (
                float(u),
                float(v),
                float(z_pnp),
                z_d,
                float(pb[0]),
                float(pb[1]),
                float(pb[2]),
            )
        )
        if z_d is None:
            cv2.circle(vis, (int(round(u)), int(round(v))), 6, (0, 0, 255), 2)
        else:
            cv2.circle(vis, (int(round(u)), int(round(v))), 4, (0, 255, 255), -1)

    diffs = np.array([z_d - z_p for _, _, z_p, z_d, *_ in rows if z_d is not None])
    z_base = np.array([r[6] for r in rows])
    n_ok = int(diffs.size)
    line = f"inliers={n_in}  rms={rms:.2f}px  base z={z_base.mean():.1f} mm"
    if n_ok:
        line += f"  depth-pnp={diffs.mean():+.1f}"
    return vis, {
        "rows": rows,
        "diffs": diffs,
        "z_base": z_base,
        "rms": rms,
        "n_in": n_in,
    }, line


def print_snapshot(stats: dict) -> None:
    diffs = stats["diffs"]
    z_base = stats["z_base"]
    print(
        f"\nSPACE  rms={stats['rms']:.3f}px  inliers={stats['n_in']}  "
        f"유효 {len(diffs)}/{len(stats['rows'])}"
    )
    print(
        f"{'u':>6} {'v':>6} {'Z_pnp':>8} {'Z_depth':>8} {'diff':>8} "
        f"{'x_base':>8} {'y_base':>8} {'z_base':>8}"
    )
    for u, v, z_p, z_d, xb, yb, zb in stats["rows"]:
        d = "—" if z_d is None else f"{z_d:8.1f}"
        df = "—" if z_d is None else f"{z_d - z_p:+8.1f}"
        print(
            f"{u:6.1f} {v:6.1f} {z_p:8.1f} {d:>8} {df:>8} "
            f"{xb:8.1f} {yb:8.1f} {zb:8.1f}"
        )
    print(
        f"평균 z_base (PnP→T_base_cam) = {z_base.mean():.2f} mm  "
        f"min {z_base.min():.1f}  max {z_base.max():.1f}"
    )
    if len(diffs):
        print(
            f"평균 depth−pnp = {diffs.mean():+.2f} mm  "
            f"std {diffs.std():.2f}"
        )
    print("책상이면 z_base ≈ 로봇 책상(약 -2 mm). -13 근처면 카메라 높이가 낮음.")


def main() -> None:
    K_raw, dist_raw, k_path = load_intrinsics()
    K, dist = intrinsics_for_rotate180(K_raw, dist_raw, FRAME_WIDTH, FRAME_HEIGHT)
    T_bc, t_path = load_T_base_cam()
    board = make_board()
    cam = OrbbecV1(FRAME_WIDTH, FRAME_HEIGHT)
    print(f"K {k_path}  180° cx={K[0, 2]:.1f} cy={K[1, 2]:.1f}")
    print(f"T_base_cam {t_path}  cam z={T_bc[2, 3]:.1f} mm")
    print("보드를 책상에 평평히. SPACE=숫자  q=종료. Viewer는 끄세요.")
    try:
        while True:
            bgr, depth = cam.grab(timeout_ms=800)
            if bgr is None or depth is None:
                continue
            if ROTATE_180:
                bgr, depth = rotate_rgbd(bgr, depth)
            vis, stats, line = compare_frame(bgr, depth, board, K, dist, T_bc)
            depth_vis = _colorize_depth(depth)
            cv2.putText(
                vis,
                line[:90],
                (8, vis.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if stats is not None and len(stats["diffs"]) else (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WIN, vis)
            cv2.imshow("depth", depth_vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord(" "),) and stats is not None:
                print_snapshot(stats)
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
