#!/usr/bin/env python3
"""저장된 손-눈 샘플 → T_base_cam (mm). 카메라는 열지 않는다.

캡처는 `handeye/capture.py`. 이 스크립트는 png+json 만 읽고 Park 등을 푼다.

  python vision/handeye/compute.py
  python vision/handeye/compute.py --align-desk
           빈 책상 RANSAC으로 T_base_cam 회전만 로봇 XY에 맞춤.
           Viewer·roi_cloud 끄고, 로봇은 책상에 평평히.
  python vision/handeye/compute.py --align-desk --desk-plane=a,b,c,d
           카메라 없이. Open3D --desk-plane (베이스) 값을 그대로.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from motion.base_frame import USER_BASE_FRAME, T_user_from_stored

from vision.calib import (
    CALIB_DIR,
    load_T_base_cam,
    load_intrinsics,
    save_handeye,
    intrinsics_for_rotate180,
)
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH
from vision.handeye.charuco import (
    MARKER_MM,
    MIN_CORNERS,
    SQUARE_MM,
    SQUARES_X,
    SQUARES_Y,
    detect_board,
    make_board,
    match_points,
)
from vision.transforms import (
    align_T_base_cam_to_base_desk_plane,
    align_T_base_cam_to_desk_z,
    invert_T,
    plane_tilt_from_z_deg,
    rt_to_T,
    transform_plane,
)

SAMPLES_DIR = CALIB_DIR / "handeye_tcp"
GRIPPER_FRAME = "tcp"
MIN_SOLVE = 12
RECOMMENDED = 20
PNP_RMS_MAX_PX = 0.8

_HANDEYE_METHODS = (
    ("park", "CALIB_HAND_EYE_PARK"),
    ("tsai", "CALIB_HAND_EYE_TSAI"),
    ("horaud", "CALIB_HAND_EYE_HORAUD"),
    ("andreff", "CALIB_HAND_EYE_ANDREFF"),
    ("daniilidis", "CALIB_HAND_EYE_DANIILIDIS"),
)


def list_samples(samples_dir: Path) -> list[Path]:
    return sorted(samples_dir.glob("*.png"))


def calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, method=None):
    fn = getattr(cv2, "calibrateHandEye", None)
    if fn is not None:
        kw = {}
        if method is not None:
            kw["method"] = method
        else:
            kw["method"] = cv2.CALIB_HAND_EYE_PARK
        return fn(R_g2b, t_g2b, R_t2c, t_t2c, **kw)
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


def solve_eye_to_hand(T_base_tcp_list, T_cam_board_list, method=None) -> np.ndarray:
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
    R, t = calibrate_hand_eye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    return rt_to_T(R, t)


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
        T_base_tcp_list.append(
            T_user_from_stored(np.asarray(meta["T_base_tcp"], dtype=float), meta.get("base_frame"))
        )
        reprojection_rms.append(rms)
        names.append(png.name)
        print(f"ok {png.name} corners={det.n_corners} inliers={n_inlier} rms={rms:.3f}px")

    n_detected = len(names)
    if n_detected < MIN_SOLVE:
        raise RuntimeError(
            f"유효 샘플 {n_detected}장 — {RECOMMENDED}장 정도를 다시 찍으세요 (최소 {MIN_SOLVE})"
        )

    def calibrate_indices(indices, method=None):
        return solve_eye_to_hand(
            [T_base_tcp_list[i] for i in indices],
            [T_cam_board_list[i] for i in indices],
            method=method,
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

    method_name = "park"
    T_base_cam = calibrate_indices(keep)
    if getattr(cv2, "calibrateHandEye", None) is not None:
        best_score = None
        for name, attr in _HANDEYE_METHODS:
            flag = getattr(cv2, attr, None)
            if flag is None:
                continue
            try:
                T = calibrate_indices(keep, flag)
            except Exception as exc:
                print(f"hand-eye {name} 실패: {exc}")
                continue
            _, _, te, re = tcp_board_residuals(T, keep)
            score = float(te.mean()) + 2.0 * float(re.mean())
            print(
                f"hand-eye {name}: residual mm {te.mean():.2f}  deg {re.mean():.3f}"
            )
            if best_score is None or score < best_score:
                best_score = score
                T_base_cam = T
                method_name = name
        print("선택", method_name)
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
        "base_frame": USER_BASE_FRAME,
        "gripper_frame": GRIPPER_FRAME,
        "unit": "mm",
        "method": f"charuco_ransac_pnp_handeye_{method_name}_eye_to_hand",
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
    print("책상이 로봇 XY와 안 맞으면 (빈 책상, Viewer 끄고):")
    print("  python vision/handeye/compute.py --align-desk")
    return T_base_cam


def _mean_plane(planes: list[np.ndarray]) -> np.ndarray:
    ns = []
    ds = []
    ref = None
    for plane in planes:
        n = np.asarray(plane[:3], dtype=np.float64).reshape(3)
        length = float(np.linalg.norm(n))
        n = n / max(length, 1e-12)
        d = float(plane[3]) / max(length, 1e-12)
        if ref is None:
            ref = n
        if float(n @ ref) < 0.0:
            n = -n
            d = -d
        ns.append(n)
        ds.append(d)
    n = np.mean(np.stack(ns, axis=0), axis=0)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return np.array([n[0], n[1], n[2], float(np.mean(ds))], dtype=np.float64)


def grab_desk_plane_cam(*, frames: int = 8) -> np.ndarray:
    from vision.calib import load_K, intrinsics_for_rotate180
    from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180
    from yolo.pose.depth_cloud import fit_plane_ransac, open_orbbec, points_from_mask, rotate180

    K, _ = load_K()
    if ROTATE_180:
        K, _ = intrinsics_for_rotate180(K, None, FRAME_WIDTH, FRAME_HEIGHT)
    cam = open_orbbec()
    planes = []
    try:
        for i in range(max(3, int(frames))):
            bgr, depth = cam.grab(timeout_ms=800)
            if bgr is None or depth is None:
                print(f"desk frame {i + 1}: no RGB-D")
                continue
            if ROTATE_180:
                bgr, depth = rotate180(bgr, depth)
            if depth.shape[:2] != bgr.shape[:2]:
                depth = cv2.resize(
                    depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                )
            h, w = depth.shape[:2]
            xyz, _ = points_from_mask(depth, K, np.ones((h, w), dtype=bool), stride=5)
            try:
                plane, inliers = fit_plane_ransac(xyz)
            except ValueError as exc:
                print(f"desk frame {i + 1}: {exc}")
                continue
            planes.append(plane)
            print(f"desk frame {i + 1} inliers={inliers}")
            time.sleep(0.05)
    finally:
        cam.close()
    if len(planes) < 3:
        raise RuntimeError(
            "책상 평면이 부족합니다. Viewer·roi_cloud 를 끄고 빈 책상을 보여 주세요."
        )
    return _mean_plane(planes)


def _parse_plane4(text: str) -> np.ndarray:
    vals = [float(x) for x in str(text).replace(",", " ").split()]
    if len(vals) != 4:
        raise SystemExit(f"--desk-plane 은 a,b,c,d 4개여야 합니다. 지금 {len(vals)}개")
    return np.asarray(vals, dtype=np.float64)


def _write_aligned(path: Path, T_new: np.ndarray, tilt_old: float, tilt_new: float, z0: float) -> Path:
    data = json.loads(path.read_text(encoding="utf-8"))
    data["T_base_cam"] = T_new.tolist()
    data["t_base_cam_mm"] = T_new[:3, 3].tolist()
    data["desk_align_tilt_deg_before"] = float(tilt_old)
    data["desk_align_tilt_deg_after"] = float(tilt_new)
    data["desk_align_z_at_origin_mm"] = float(z0)
    data["method"] = str(data.get("method", "handeye")) + "_desk_z_align"
    dest = save_handeye(data, path)
    print("저장", dest)
    return dest


def align_desk(desk_plane_base: np.ndarray | None = None) -> np.ndarray:
    T_old, path = load_T_base_cam()
    print("T", path)
    if desk_plane_base is not None:
        plane_base = np.asarray(desk_plane_base, dtype=np.float64).reshape(4)
        print("카메라 없이 --desk-plane (베이스 프레임) 사용")
        T_new = align_T_base_cam_to_base_desk_plane(T_old, plane_base)
        plane_new = transform_plane(transform_plane(plane_base, invert_T(T_old)), T_new)
        tilt_old = plane_tilt_from_z_deg(plane_base)
    else:
        print("빈 책상, 물체 치우기. Viewer·roi_cloud 끄기.")
        try:
            plane_cam = grab_desk_plane_cam()
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}\n카메라가 없으면 방금 Open3D --desk-plane 값으로:\n"
                "  python vision/handeye/compute.py --align-desk --desk-plane=a,b,c,d"
            ) from exc
        plane_base = transform_plane(plane_cam, T_old)
        tilt_old = plane_tilt_from_z_deg(plane_base)
        T_new = align_T_base_cam_to_desk_z(T_old, plane_cam)
        plane_new = transform_plane(plane_cam, T_new)
    tilt_new = plane_tilt_from_z_deg(plane_new)
    n = np.asarray(plane_new[:3], dtype=np.float64)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    z0 = -float(plane_new[3]) / n[2] if abs(float(n[2])) > 1e-6 else float("nan")
    print(f"책상 vs 로봇 XY  {tilt_old:.2f}° → {tilt_new:.2f}°")
    print("카메라 원점 in base mm", np.round(T_new[:3, 3], 2))
    print(f"맞춘 뒤 원점에서 책상 z={z0:.2f} mm")
    _write_aligned(path, T_new, tilt_old, tilt_new, z0)
    return T_new


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-ARM eye-to-hand compute (saved samples → T_base_cam)")
    p.add_argument("--samples-dir", type=Path, default=SAMPLES_DIR)
    p.add_argument(
        "--align-desk",
        action="store_true",
        help="빈 책상 RANSAC으로 T_base_cam 회전을 로봇 XY에 맞춤",
    )
    p.add_argument(
        "--desk-plane",
        default=None,
        help="베이스 프레임 ax,by,cz,d. 있으면 카메라 없이 --align-desk",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples_dir = args.samples_dir
    samples_dir.mkdir(parents=True, exist_ok=True)

    if args.align_desk:
        plane = _parse_plane4(args.desk_plane) if args.desk_plane else None
        align_desk(plane)
        return

    n = len(list_samples(samples_dir))
    if n < MIN_SOLVE:
        raise SystemExit(
            f"샘플 {n}장. 최소 {MIN_SOLVE}장. 먼저 python vision/handeye/capture.py"
        )
    compute(samples_dir)


if __name__ == "__main__":
    main()
