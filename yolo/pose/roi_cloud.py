#!/usr/bin/env python3
"""YOLO ROI → 인스턴스 점군 → 최상단 → (선택) CAD 등록.

V4L2 컬러만으로는 안 된다. Orbbec SDK(D2C) + 학습한 best.pt / best-seg.pt.
펜던트·Viewer·detect.py 와 동시에 켜지 말 것.

  python yolo/pose/roi_cloud.py
  python yolo/pose/roi_cloud.py --mask          # 세그 ROI + RGB 색칠(뎁스는 마스킹 없음)
  python yolo/pose/roi_cloud.py --mask --cad    # 사진에 CAD 축 + xyzrpy
  python yolo/pose/roi_cloud.py --mask --cad --base
  python yolo/pose/roi_cloud.py --mask --no-noise-filter
  v=3D  c=지금 등록  s=PLY  r=평면  q=종료
  별도 창: NoiseRemoval / HoleFilter / HoleFilling / Color / DepthExp / Temporal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vision.calib import intrinsics_for_rotate180, load_K, load_T_base_cam
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180
from vision.transforms import invert_T, plane_tilt_from_z_deg, transform_plane
from vision.orbbec_filters import (
    NOISE_MAX_SIZE_DEFAULT,
    NOISE_MIN_DIFF_DEFAULT,
    OrbbecFilterPanel,
)
from yolo.config import BEST_PT, BEST_SEG_PT, DETECT_CONF, RUNS_DIR, add_class_argument, class_from_args, quiet_gtk
from yolo.pose.depth_cloud import (
    colorize_depth,
    fit_plane_ransac,
    open_orbbec,
    points_from_mask,
    rotate180,
    write_ply,
)
from yolo.pose.debug.local_plane import SLICE_MM, local_pose
from yolo.pose.instances import MASK_PLANE_MM, collect_instances, select_topmost

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WIN = "YOLO box ROI cloud"
ROI_DIR = RUNS_DIR / "roi"


def _work_desk_plane(plane, T_bc) -> np.ndarray | None:
    """등록 프레임의 책상. --base 면 로봇 XY, 없으면 카메라 RANSAC."""
    if plane is not None:
        p = np.asarray(plane, dtype=np.float64).reshape(4)
        if T_bc is not None:
            p = transform_plane(p, T_bc)
        return p
    if T_bc is not None:
        return np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    return None


def _pose_mm_to_T(xyz_mm, rpy_deg) -> np.ndarray:
    from motion.robot_kinematics import rpy_deg_to_rotmat

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_deg_to_rotmat(
        float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2])
    )
    T[:3, 3] = np.asarray(xyz_mm, dtype=np.float64).reshape(3)
    return T


def _load_grasp_cad() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """model.yaml `grasp` → T_cad_grasp(mm), xyz, rpy. 티칭 물체 기준 좌표계."""
    import ast

    from yolo.config import CAD_YAML

    xyz = np.zeros(3, dtype=np.float64)
    rpy = np.zeros(3, dtype=np.float64)
    if CAD_YAML.is_file():
        section = None
        for raw in CAD_YAML.read_text(encoding="utf-8").splitlines():
            if "#" in raw:
                raw = raw.split("#", 1)[0]
            if not raw.strip():
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            line = raw.strip()
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key, value = key.strip(), value.strip()
            if indent == 0:
                section = key if value == "" else None
                continue
            if section != "grasp" or not value:
                continue
            parsed = np.asarray(ast.literal_eval(value), dtype=np.float64).reshape(-1)
            if key == "xyz_mm" and parsed.size >= 3:
                xyz = parsed[:3].copy()
            elif key == "rpy_deg" and parsed.size >= 3:
                rpy = parsed[:3].copy()
    return _pose_mm_to_T(xyz, rpy), xyz, rpy


def _tcp_from_cad(T_work_cad: np.ndarray, T_cad_grasp: np.ndarray):
    """T_work_grasp = T_work_cad @ T_cad_grasp → xyz mm, rpy deg."""
    from yolo.pose.register import T_to_xyzrpy

    Tg = np.asarray(T_cad_grasp, dtype=np.float64).reshape(4, 4)
    Tw = np.asarray(T_work_cad, dtype=np.float64).reshape(4, 4)
    if np.allclose(Tg, np.eye(4), atol=1e-9):
        return T_to_xyzrpy(Tw)
    return T_to_xyzrpy(Tw @ Tg)


def _xyz_in_cad(T_work_cad: np.ndarray, xyz_work: np.ndarray) -> np.ndarray:
    """작업 프레임 점 → 물체(CAD) 기준 좌표."""
    T_cw = invert_T(np.asarray(T_work_cad, dtype=np.float64).reshape(4, 4))
    p = np.asarray(xyz_work, dtype=np.float64).reshape(3)
    return T_cw[:3, :3] @ p + T_cw[:3, 3]


def _place_ui_xyzrpy(T_work_cad: np.ndarray):
    """펜던트 place UI용: xyz + 베이스 RPY + 물체축 RPY.

    베이스 숫자는 물체축→extrinsic 으로 고정(canonicalize)해서
    펜던트에 물체 RPY를 넣었을 때와 같은 베이스 칸 값이 나오게 한다.
    """
    from pendant.teach_grasp import (
        canonicalize_extrinsic_rpy,
        rpy_extrinsic_to_body_xyz,
    )
    from yolo.pose.register import T_to_xyzrpy

    xyz, rpy_raw = T_to_xyzrpy(np.asarray(T_work_cad, dtype=np.float64).reshape(4, 4))
    rpy_base = canonicalize_extrinsic_rpy(rpy_raw)
    rpy_body = rpy_extrinsic_to_body_xyz(rpy_base)
    return (
        np.asarray(xyz, dtype=float).reshape(3),
        np.asarray(rpy_base, dtype=float).reshape(3),
        np.asarray(rpy_body, dtype=float).reshape(3),
    )


def _project_cam(xyz: np.ndarray, K: np.ndarray) -> tuple[int, int] | None:
    z = float(xyz[2])
    if z < 20.0:
        return None
    u = float(K[0, 0]) * float(xyz[0]) / z + float(K[0, 2])
    v = float(K[1, 1]) * float(xyz[1]) / z + float(K[1, 2])
    return int(round(u)), int(round(v))


def _draw_cad_axes(
    img: np.ndarray,
    T_cam: np.ndarray,
    K: np.ndarray,
    *,
    length_mm: float = 30.0,
) -> None:
    """카메라 mm 의 T_cam_cad 를 사진에 RGB 축으로 그린다."""
    T = np.asarray(T_cam, dtype=np.float64).reshape(4, 4)
    origin = T[:3, 3]
    po = _project_cam(origin, K)
    if po is None:
        return
    h, w = img.shape[:2]
    if not (0 <= po[0] < w and 0 <= po[1] < h):
        return
    colors_bgr = ((0, 0, 255), (0, 255, 0), (255, 80, 80))
    for i, color in enumerate(colors_bgr):
        tip = origin + T[:3, i] * length_mm
        pt = _project_cam(tip, K)
        if pt is None:
            continue
        cv2.arrowedLine(img, po, pt, color, 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(img, po, 4, (255, 255, 255), -1, cv2.LINE_AA)


def _median_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.median(np.asarray(xyz, dtype=np.float64).reshape(-1, 3), axis=0)


def _device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


class _CadWorker:
    """등록은 수 초 걸리므로 카메라 루프 밖에서 돌린다."""

    def __init__(
        self,
        *,
        voxel_mm: float,
        flip: bool,
        tries: int,
        camera_origin: np.ndarray | None = None,
        T_cad_grasp: np.ndarray | None = None,
        grasp_xyz_mm: np.ndarray | None = None,
        grasp_rpy_deg: np.ndarray | None = None,
        frame_name: str = "camera_mm",
    ):
        from yolo.pose.register import load_cad_xyz, register_pose

        self._register_pose = register_pose
        self.voxel_mm = voxel_mm
        self.flip = flip
        self.tries = tries
        self.camera_origin = (
            None
            if camera_origin is None
            else np.asarray(camera_origin, dtype=np.float64).reshape(3)
        )
        self.T_cad_grasp = (
            np.eye(4, dtype=np.float64)
            if T_cad_grasp is None
            else np.asarray(T_cad_grasp, dtype=np.float64).reshape(4, 4)
        )
        self.grasp_xyz_mm = (
            np.zeros(3, dtype=np.float64)
            if grasp_xyz_mm is None
            else np.asarray(grasp_xyz_mm, dtype=np.float64).reshape(3)
        )
        self.grasp_rpy_deg = (
            np.zeros(3, dtype=np.float64)
            if grasp_rpy_deg is None
            else np.asarray(grasp_rpy_deg, dtype=np.float64).reshape(3)
        )
        self.frame_name = frame_name
        self._lock = threading.Lock()
        self._job: np.ndarray | None = None
        self._T_init: np.ndarray | None = None
        self._T_prev: np.ndarray | None = None
        self._desk_plane: np.ndarray | None = None
        self.result: dict | None = None
        self.error: str | None = None
        self.busy = False
        self._stop = False
        load_cad_xyz(voxel_mm=voxel_mm)
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def submit(
        self,
        xyz: np.ndarray,
        *,
        T_init: np.ndarray | None = None,
        T_prev: np.ndarray | None = None,
        desk_plane: np.ndarray | None = None,
    ) -> bool:
        if len(xyz) < 20:
            return False
        with self._lock:
            if self.busy:
                return False
            self._job = np.asarray(xyz, dtype=np.float64).copy()
            self._T_init = None if T_init is None else np.asarray(T_init, dtype=np.float64).copy()
            self._T_prev = None if T_prev is None else np.asarray(T_prev, dtype=np.float64).copy()
            self._desk_plane = (
                None
                if desk_plane is None
                else np.asarray(desk_plane, dtype=np.float64).reshape(4).copy()
            )
            self.busy = True
            self.error = None
        return True

    def _loop(self) -> None:
        while not self._stop:
            with self._lock:
                xyz = self._job
                T_init = self._T_init
                T_prev = self._T_prev
                desk_plane = self._desk_plane
                self._job = None
            if xyz is None:
                time.sleep(0.03)
                continue
            try:
                pose = self._register_pose(
                    xyz,
                    T_base_cam=None,
                    voxel_mm=self.voxel_mm,
                    flip=self.flip,
                    tries=self.tries,
                    T_init=T_init,
                    T_prev=T_prev,
                    camera_origin=self.camera_origin,
                    desk_plane=desk_plane,
                )
                err = None
            except Exception as exc:
                pose = None
                err = str(exc)
            with self._lock:
                if pose is not None:
                    self.result = pose
                self.error = err
                self.busy = False
            if pose is not None:
                xyz_mm = pose["xyz_mm"]
                rpy = pose["rpy_deg"]
                kind = "추적" if str(pose.get("source", "")).endswith("track") else "등록"
                ok = "ok" if pose.get("aligned_ok") else "miss"
                print(
                    f"CAD {kind} {ok} {self.frame_name} "
                    f"xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
                    f"rpy_ext=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]  "
                    f"on={pose.get('scene_frac', 0):.2f} "
                    f"vis={pose.get('vis_frac', 0):.2f} "
                    f"iou={pose.get('xy_iou', 0):.2f}"
                )
                g = self.grasp_xyz_mm
                gr = self.grasp_rpy_deg
                print(
                    f"  grasp(티칭/물체축) xyz=[{g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}]  "
                    f"rpy=[{gr[0]:.1f}, {gr[1]:.1f}, {gr[2]:.1f}]"
                )
                if pose.get("aligned_ok", False):
                    p_xyz, p_base, p_body = _place_ui_xyzrpy(pose["T"])
                    print(
                        f"  물체 x 축: {p_xyz[0]:.1f} (mm),  y 축: {p_xyz[1]:.1f} (mm),  "
                        f"z 축: {p_xyz[2]:.1f} (mm)"
                    )
                    print(
                        f"  물체 Rx: {p_body[0]:.1f} ,  Ry: {p_body[1]:.1f} ,  Rz: {p_body[2]:.1f}"
                    )
                    print(
                        f"  베이스 기준 Rx: {p_base[0]:.1f} ,  Ry: {p_base[1]:.1f} ,  "
                        f"Rz: {p_base[2]:.1f}"
                    )
            elif err:
                print(f"CAD 등록 실패: {err}")

    def close(self) -> None:
        self._stop = True

    def take_result(self) -> dict | None:
        """완료 결과를 한 번만 소비한다."""
        with self._lock:
            result = self.result
            self.result = None
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO box → RGB-D point ROI")
    parser.add_argument(
        "--mask",
        action="store_true",
        help="세그 마스크로 ROI + RGB만 색칠(박스 없음). 뎁스는 원본",
    )
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=DETECT_CONF)
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="장면(회색) 픽셀 간격. 1이면 전 픽셀",
    )
    parser.add_argument("--pad", type=int, default=2)
    parser.add_argument(
        "--show-obb",
        action="store_true",
        help="RGB·뎁스에 YOLO OBB 네 변을 그림 (자홍). 노란 선은 ROI 윤곽",
    )
    parser.add_argument(
        "--plane-mm",
        type=float,
        default=4.0,
        help="박스 모드에서 책상 평면 이내 점 제거(mm). --mask 는 2mm 고정",
    )
    parser.add_argument("--no-plane", action="store_true", help="RANSAC 책상 제거 안 함")
    parser.add_argument(
        "--slice-mm",
        type=float,
        default=SLICE_MM,
        help="윗면 슬라이스 두께(mm). 옆면 제거",
    )
    parser.add_argument(
        "--depth-span",
        type=float,
        default=60.0,
        help="오른쪽 뎁스 색상 범위(mm). 작을수록 대비가 강함",
    )
    parser.add_argument("--no-rotate-180", action="store_true")
    parser.add_argument("--base", action="store_true", help="점군을 베이스 좌표(mm)로")
    parser.add_argument(
        "--cad",
        action="store_true",
        help="최상단 ROI vs CAD 등록. 화면에 xyzrpy",
    )
    parser.add_argument("--voxel-mm", type=float, default=1.0, help="등록용 다운샘플(mm)")
    parser.add_argument(
        "--overlay-voxel-mm",
        type=float,
        default=0.5,
        help="v 키 빨간 CAD 간격(mm)",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="stride=1, 등록 1mm, 빨간 CAD 0.5mm (지금 기본값)",
    )
    parser.add_argument(
        "--cad-every",
        type=float,
        default=2.5,
        metavar="SEC",
        help="첫 등록 재시도·물체 이동 후 ICP 주기(초). c 는 전체 RANSAC",
    )
    parser.add_argument(
        "--cad-move-mm",
        type=float,
        default=8.0,
        metavar="MM",
        help="점군 중심이 이만큼 움직여야 ICP를 다시 함. 정지 시 축 고정",
    )
    parser.add_argument("--intrinsics", type=Path, default=None)
    parser.add_argument(
        "--no-noise-filter",
        action="store_true",
        help="Orbbec NoiseRemovalFilter 끄기. 기본은 켜짐",
    )
    parser.add_argument(
        "--noise-min-diff",
        type=int,
        default=None,
        metavar="N",
        help="Viewer Min Diff. 기본 10000",
    )
    parser.add_argument(
        "--noise-max-size",
        type=int,
        default=None,
        metavar="N",
        help="Viewer Max Size. 기본 1",
    )
    parser.add_argument(
        "--no-outlier",
        action="store_true",
        help="v 키 Open3D에서 회색 outlier 제거 끄기",
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=1.5,
        help="v 키 회색 outlier. 작을수록 더 지움",
    )
    parser.add_argument("--outlier-nb", type=int, default=20)
    add_class_argument(parser)
    args = parser.parse_args()
    if args.dense:
        args.stride = 1
        args.voxel_mm = min(float(args.voxel_mm), 1.0)
        args.overlay_voxel_mm = min(float(args.overlay_voxel_mm), 0.5)
    class_name = class_from_args(args)

    rotate = ROTATE_180 and not args.no_rotate_180
    weights = args.weights if args.weights is not None else (
        BEST_SEG_PT if args.mask else BEST_PT
    )
    if not Path(weights).exists():
        hint = "python yolo/train/train.py --seg" if args.mask else "python yolo/train/train.py"
        raise SystemExit(f"가중치 없음: {weights}\n{hint}")

    K, k_path = load_K(args.intrinsics)
    if rotate:
        K, _ = intrinsics_for_rotate180(K, None, FRAME_WIDTH, FRAME_HEIGHT)
        print(
            f"K {k_path}  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}  "
            f"cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} (180°)"
        )
    else:
        print(f"K {k_path}  fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}")
    if K[0, 0] > 2000:
        print("주의: fx가 해상도 640에 비해 큽니다. 렌즈 캘리브를 다시 보는 편이 좋습니다.")

    T_bc = None
    frame_name = "camera_mm"
    if args.base:
        T_bc, t_path = load_T_base_cam()
        frame_name = "base_mm"
        print(f"T_base_cam {t_path}")

    noise_min_diff = (
        args.noise_min_diff if args.noise_min_diff is not None else NOISE_MIN_DIFF_DEFAULT
    )
    noise_max_size = (
        args.noise_max_size if args.noise_max_size is not None else NOISE_MAX_SIZE_DEFAULT
    )

    # 카메라보다 먼저 로드. 스트림 중 긴 로드는 OpenNI 큐 폭주·USB 단절을 부른다.
    print("YOLO 로드 중...")
    from ultralytics import YOLO

    device = _device()
    model = YOLO(str(weights))

    keys = "v=3D  s=save  r=plane  q=quit"
    if args.cad:
        keys = "c=등록  " + keys
    print(f"class={class_name}  device={device}  {keys}")
    last_xyz = np.zeros((0, 3), dtype=np.float32)
    last_rgb = np.zeros((0, 3), dtype=np.uint8)
    plane = None
    refit_plane = not args.no_plane
    last_pose = None
    cad_pose = None
    viewer_process = None
    cad_worker = None
    last_cad_t = 0.0
    last_cad_anchor = None
    if args.cad:
        print("CAD 로드 중...")
        T_cad_grasp, grasp_xyz_mm, grasp_rpy_deg = _load_grasp_cad()
        print(
            f"obj(티칭) xyz=[{grasp_xyz_mm[0]:.1f}, {grasp_xyz_mm[1]:.1f}, {grasp_xyz_mm[2]:.1f}]  "
            f"rpy=[{grasp_rpy_deg[0]:.1f}, {grasp_rpy_deg[1]:.1f}, {grasp_rpy_deg[2]:.1f}]"
        )
        cad_worker = _CadWorker(
            voxel_mm=args.voxel_mm,
            flip=bool(args.base),
            tries=4,
            camera_origin=None if T_bc is None else T_bc[:3, 3],
            T_cad_grasp=T_cad_grasp,
            grasp_xyz_mm=grasp_xyz_mm,
            grasp_rpy_deg=grasp_rpy_deg,
            frame_name=frame_name,
        )
        print(
            "CAD: 첫 등록 후 정지하면 축 고정. "
            f"점군이 {args.cad_move_mm:.0f}mm 이상 움직이면 ICP. c 는 처음부터."
        )
        print(
            f"밀도  stride={args.stride}  등록={args.voxel_mm:g}mm  "
            f"표시={args.overlay_voxel_mm:g}mm"
        )
    else:
        T_cad_grasp = np.eye(4, dtype=np.float64)
        grasp_xyz_mm = np.zeros(3, dtype=np.float64)
        grasp_rpy_deg = np.zeros(3, dtype=np.float64)

    cam = open_orbbec(
        noise_filter=not args.no_noise_filter,
        noise_min_diff=noise_min_diff,
        noise_max_size=noise_max_size,
        hole_filter=True,
    )
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    filters = OrbbecFilterPanel(
        cam,
        noise_on=not args.no_noise_filter,
        min_diff=noise_min_diff,
        max_size=noise_max_size,
    )

    try:
        while True:
            cam.flush()
            bgr, depth = cam.grab(timeout_ms=800)
            if bgr is None or depth is None:
                vis = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(vis, "no RGB-D", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(WIN, vis)
                filters.sync()
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue
            if rotate:
                bgr, depth = rotate180(bgr, depth)
            if depth.shape[:2] != bgr.shape[:2]:
                depth = cv2.resize(depth, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

            h, w = bgr.shape[:2]
            if not args.no_plane and refit_plane:
                full = np.ones((h, w), dtype=bool)
                table_xyz, _ = points_from_mask(depth, K, full, stride=5)
                try:
                    plane, inliers = fit_plane_ransac(table_xyz)
                    extra = ""
                    if T_bc is not None:
                        desk = transform_plane(plane, T_bc)
                        extra = (
                            f"  vs 로봇 XY {plane_tilt_from_z_deg(desk):.2f}°"
                        )
                    print(
                        "책상 평면 "
                        f"[{plane[0]:.4f}, {plane[1]:.4f}, {plane[2]:.4f}, "
                        f"{plane[3]:.1f}]  inliers={inliers}{extra}"
                    )
                except ValueError as exc:
                    plane = None
                    print(f"책상 평면 실패: {exc}. r로 다시 시도하세요.")
                refit_plane = False

            results = model.predict(
                bgr, conf=args.conf, device=device, verbose=False, imgsz=640
            )
            result = results[0]
            depth_vis = colorize_depth(depth, center_span_mm=args.depth_span)
            instances = collect_instances(
                depth=depth,
                bgr=bgr,
                K=K,
                result=result,
                plane=plane,
                use_mask=args.mask,
                pad=args.pad,
                plane_mm=args.plane_mm,
                stride=args.stride,
                T_base_cam=T_bc,
            )
            chosen = select_topmost(instances)
            n_det = len(instances)
            last_pose = None
            if chosen is None:
                last_xyz = np.zeros((0, 3), dtype=np.float32)
                last_rgb = np.zeros((0, 3), dtype=np.uint8)
            else:
                pick = instances[chosen]
                last_xyz, last_rgb = pick.xyz, pick.rgb
                last_pose = local_pose(
                    pick.xyz,
                    desk_plane=None if T_bc is not None else plane,
                    band_mm=args.slice_mm,
                    in_base=T_bc is not None,
                )

            if cad_worker is not None:
                nxt = cad_worker.take_result()
                if nxt is not None:
                    # 실패한 후보는 숫자·축·Open3D에 절대 내보내지 않는다.
                    if nxt.get("aligned_ok", False):
                        cad_pose = nxt
                        if len(last_xyz) >= 20:
                            last_cad_anchor = _median_xyz(last_xyz)
                    else:
                        cad_pose = None
                        last_cad_anchor = None
                        # miss 직후 due 가 이미 지나 있어 바로 전체 RANSAC이 다시 돈다.
                        last_cad_t = time.monotonic()
                now = time.monotonic()
                due = (now - last_cad_t) >= max(0.2, float(args.cad_every))
                pose_ok = cad_pose is not None and cad_pose.get("aligned_ok", False)
                if len(last_xyz) >= 20 and due:
                    T_keep = cad_pose["T"] if pose_ok else None
                    anchor = _median_xyz(last_xyz)
                    moved = (
                        last_cad_anchor is None
                        or float(np.linalg.norm(anchor - last_cad_anchor))
                        >= max(1.0, float(args.cad_move_mm))
                    )
                    if not pose_ok or moved:
                        if cad_worker.submit(
                            last_xyz,
                            T_init=T_keep,
                            T_prev=T_keep,
                            desk_plane=_work_desk_plane(plane, T_bc),
                        ):
                            last_cad_t = now
                            if pose_ok:
                                last_cad_anchor = anchor

            for i, inst in enumerate(instances):
                x1, y1, x2, y2 = [int(v) for v in inst.xyxy]
                color = (0, 255, 255) if i == chosen else (0, 180, 0)
                if args.show_obb and inst.quad is not None:
                    obb_pts = np.round(inst.quad).astype(np.int32).reshape(-1, 1, 2)
                    obb_color = (255, 0, 255) if i == chosen else (180, 0, 180)
                    cv2.polylines(bgr, [obb_pts], True, obb_color, 1, cv2.LINE_AA)
                    cv2.polylines(depth_vis, [obb_pts], True, obb_color, 1, cv2.LINE_AA)
                if args.mask:
                    tint = np.zeros_like(bgr)
                    tint[inst.roi] = color
                    cv2.addWeighted(tint, 0.35, bgr, 1.0, 0.0, dst=bgr)
                    ys, xs = np.nonzero(inst.roi)
                    if len(xs):
                        tag_x = int(xs.min())
                        tag_y = max(16, int(ys.min()) - 6)
                    else:
                        tag_x, tag_y = x1, max(16, y1 - 6)
                else:
                    tag_x, tag_y = x1, max(16, y1 - 6)
                tag = f"{len(inst.xyz)} pts zok={inst.n_zok}"
                if i == chosen and last_pose is not None:
                    tag = (
                        f"yaw={last_pose['yaw_deg']:.1f} n={last_pose['n_points']}"
                        f" zok={inst.n_zok}"
                    )
                cv2.putText(
                    bgr,
                    tag,
                    (tag_x, tag_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            ztxt = ""
            if len(last_xyz):
                ztxt = f"  zmed={float(np.median(last_xyz[:, 2])):.0f}"
            yawtxt = ""
            if last_pose is not None:
                c = last_pose["center_mm"]
                yawtxt = f"  yaw={last_pose['yaw_deg']:.1f} c={c[0]:.0f},{c[1]:.0f}"
            if plane is None:
                plane_text = "plane=off"
            else:
                cut = MASK_PLANE_MM if args.mask else args.plane_mm
                plane_text = f"plane>{cut:.0f}mm"
            hint = "v=3D s=save r=plane q=quit"
            if args.cad:
                hint = "c=CAD " + hint
            cv2.putText(
                bgr,
                f"n={n_det} sel={0 if chosen is None else chosen} pts={len(last_xyz)}"
                f"{ztxt}{yawtxt} {plane_text} {frame_name} {hint}",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            if args.cad:
                place_font = 0.58
                place_thick = 2
                if cad_pose is not None:
                    on = cad_pose.get("scene_frac")
                    vis = cad_pose.get("vis_frac")
                    iou = cad_pose.get("xy_iou")
                    ok = cad_pose.get("aligned_ok", False)
                    p_xyz, p_base, p_body = _place_ui_xyzrpy(cad_pose["T"])
                    xyz_line = (
                        f"물체 x 축: {p_xyz[0]:.1f} (mm),  y 축: {p_xyz[1]:.1f} (mm),  "
                        f"z 축: {p_xyz[2]:.1f} (mm)"
                    )
                    body_line = (
                        f"물체 Rx: {p_body[0]:.1f} ,  Ry: {p_body[1]:.1f} ,  "
                        f"Rz: {p_body[2]:.1f}"
                    )
                    base_line = (
                        f"베이스 기준 Rx: {p_base[0]:.1f} ,  Ry: {p_base[1]:.1f} ,  "
                        f"Rz: {p_base[2]:.1f}"
                    )
                    cad_line = (
                        f"CAD/{frame_name} on={on:.2f}" if on is not None else f"CAD/{frame_name}"
                    )
                    if vis is not None:
                        cad_line += f" vis={vis:.2f}"
                    if iou is not None:
                        cad_line += f" iou={iou:.2f}"
                    if not ok:
                        cad_line += "  MISS"
                    obj_line = (
                        f"grasp(티칭·물체축) xyz={grasp_xyz_mm[0]:.0f},{grasp_xyz_mm[1]:.0f},{grasp_xyz_mm[2]:.0f}  "
                        f"rpy={grasp_rpy_deg[0]:.1f},{grasp_rpy_deg[1]:.1f},{grasp_rpy_deg[2]:.1f}"
                    )
                elif cad_worker is not None and cad_worker.busy:
                    xyz_line = "물체 x/y/z 축: 등록 중 ..."
                    body_line = "물체 Rx/Ry/Rz: ..."
                    base_line = "베이스 기준 Rx/Ry/Rz: ..."
                    cad_line = "CAD ..."
                    obj_line = (
                        f"grasp(티칭/물체축) xyz={grasp_xyz_mm[0]:.0f},{grasp_xyz_mm[1]:.0f},{grasp_xyz_mm[2]:.0f}  "
                        f"rpy={grasp_rpy_deg[0]:.1f},{grasp_rpy_deg[1]:.1f},{grasp_rpy_deg[2]:.1f}"
                    )
                else:
                    xyz_line = "물체 x/y/z 축: 대기 (점군 나오면 시작)"
                    body_line = "물체 Rx/Ry/Rz: 대기"
                    base_line = "베이스 기준 Rx/Ry/Rz: 대기"
                    cad_line = "CAD 대기"
                    obj_line = (
                        f"grasp(티칭/물체축) xyz={grasp_xyz_mm[0]:.0f},{grasp_xyz_mm[1]:.0f},{grasp_xyz_mm[2]:.0f}  "
                        f"rpy={grasp_rpy_deg[0]:.1f},{grasp_rpy_deg[1]:.1f},{grasp_rpy_deg[2]:.1f}"
                    )
                overlay_lines = [
                    (50, xyz_line, (0, 255, 255)),
                    (78, body_line, (0, 255, 255)),
                    (106, base_line, (0, 255, 255)),
                    (134, cad_line, (180, 180, 80)),
                    (162, obj_line, (0, 220, 255)),
                ]
                for y, text, color in overlay_lines:
                    if not text:
                        continue
                    cv2.putText(
                        bgr,
                        text,
                        (8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        place_font,
                        color,
                        place_thick,
                        cv2.LINE_AA,
                    )
            if cad_pose is not None:
                T_draw = np.asarray(cad_pose["T"], dtype=np.float64)
                if T_bc is not None:
                    T_draw = invert_T(T_bc) @ T_draw
                _draw_cad_axes(bgr, T_draw, K)
                _draw_cad_axes(depth_vis, T_draw, K)
            filters.sync()
            panel = np.hstack([bgr, depth_vis])
            cv2.imshow(WIN, panel)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("r") and not args.no_plane:
                refit_plane = True
            if key == ord("c") and cad_worker is not None:
                if len(last_xyz) < 20:
                    print("등록할 점이 없습니다.")
                elif not cad_worker.submit(
                    last_xyz,
                    T_init=None,
                    T_prev=None if cad_pose is None else cad_pose["T"],
                    desk_plane=_work_desk_plane(plane, T_bc),
                ):
                    print("이미 등록 중입니다.")
                else:
                    last_cad_t = time.monotonic()
                    last_cad_anchor = None
                    print("CAD 처음부터 다시 등록")
            if key == ord("v"):
                if len(last_xyz) == 0:
                    zok = instances[0].n_zok if instances else 0
                    if zok == 0:
                        print(
                            "표시할 점이 없습니다. 박스 안 뎁스가 0입니다 "
                            "(센서 홀). 흰 종이보다 회색 매트가 낫습니다."
                        )
                    else:
                        print(
                            f"표시할 점이 없습니다. 뎁스 {zok}px 있었으나 "
                            "마스크/연결성분에서 빠졌습니다."
                        )
                    continue
                if viewer_process is not None and viewer_process.poll() is None:
                    print("Open3D 창이 이미 열려 있습니다.")
                    continue
                ROI_DIR.mkdir(parents=True, exist_ok=True)
                preview = ROI_DIR / f".preview_{frame_name}.ply"
                if cad_pose is not None:
                    from yolo.pose.register import _overlay_xyz, load_cad_xyz

                    T_ov = cad_pose["T"]
                    cad_vis = load_cad_xyz(voxel_mm=float(args.overlay_voxel_mm))
                    cam_o = None if T_bc is None else T_bc[:3, 3]
                    o_xyz, o_rgb = _overlay_xyz(
                        last_xyz,
                        last_rgb,
                        cad_vis,
                        T_ov,
                        camera_origin=cam_o,
                    )
                    write_ply(preview, o_xyz, o_rgb)
                else:
                    write_ply(preview, last_xyz, last_rgb)
                view_cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "view" / "view_cloud.py"),
                    str(preview),
                    "--no-obb",
                ]
                if cad_pose is not None:
                    # T[0,0] 이 음수면 "--cad-T" 다음 토큰이 새 옵션으로 파싱된다.
                    view_cmd.append(
                        "--cad-T="
                        + ",".join(f"{x:.9g}" for x in np.asarray(cad_pose["T"]).reshape(-1))
                    )
                if plane is not None:
                    desk = (
                        transform_plane(plane, T_bc)
                        if T_bc is not None
                        else np.asarray(plane, dtype=np.float64)
                    )
                    view_cmd.append(
                        "--desk-plane="
                        + ",".join(f"{x:.9g}" for x in np.asarray(desk).reshape(-1))
                    )
                if args.dense or float(args.overlay_voxel_mm) <= 1.0:
                    view_cmd.extend(["--point-size", "2"])
                if args.no_outlier:
                    view_cmd.append("--no-outlier")
                else:
                    view_cmd.extend(
                        [
                            "--outlier-std",
                            str(args.outlier_std),
                            "--outlier-nb",
                            str(args.outlier_nb),
                        ]
                    )
                viewer_process = subprocess.Popen(view_cmd)
                print(f"Open3D: {preview}")
            if key == ord("s"):
                if len(last_xyz) == 0:
                    print("저장할 점이 없습니다.")
                    continue
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                out = ROI_DIR / f"roi_{stamp}_{frame_name}.ply"
                write_ply(out, last_xyz, last_rgb)
                print(f"저장 {out}  n={len(last_xyz)}")
                if last_pose is not None:
                    slice_xyz = last_pose.get("sliced_xyz")
                    if slice_xyz is not None and len(slice_xyz):
                        slice_out = out.with_name(f"{out.stem}_slice.ply")
                        write_ply(slice_out, slice_xyz)
                        print(f"저장 {slice_out}  n={len(slice_xyz)}")
                    c = last_pose["center_mm"]
                    print(
                        f"local_pose yaw={last_pose['yaw_deg']:.1f} "
                        f"center={np.round(c, 1).tolist()} mm  "
                        f"n={last_pose['n_points']}"
                    )
    finally:
        cam.close()
        if cad_worker is not None:
            cad_worker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
