#!/usr/bin/env python3
"""CAD vs 인스턴스 점군 → T, xyzrpy.

카메라·YOLO·시리얼은 열지 않는다. 정본 포즈.

  CAD (mm) + ROI 점군
    → voxel · 법선 · FPFH
    → RANSAC 초기 T
    → Point-to-Plane ICP
    → xyzrpy (카메라 또는 베이스)

  python yolo/register.py --scene yolo/runs/roi/foo.ply
  python yolo/register.py --scene yolo/runs/roi/foo.ply --base
  python yolo/register.py --cad /path/to/other.stl --scene foo.ply
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.transforms import apply_T, rt_to_T
from yolo.config import cad_mesh_path, cad_mesh_rpy_deg, cad_mesh_xyz_mm, cad_unit
from yolo.depth_cloud import write_ply

MIN_POINTS = 20
_CAD_CACHE: dict[tuple, np.ndarray] = {}


def _require_o3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "open3d가 없습니다. conda activate lerobot 한 뒤 설치하세요."
        ) from exc
    return o3d


def cad_mesh_R() -> np.ndarray:
    """model.yaml mesh_rpy → 3x3. 파일 좌표 → 프로젝트 CAD 프레임."""
    from motion.robot_kinematics import rpy_deg_to_rotmat

    r, p, y = cad_mesh_rpy_deg()
    if abs(r) < 1e-12 and abs(p) < 1e-12 and abs(y) < 1e-12:
        return np.eye(3)
    return rpy_deg_to_rotmat(r, p, y)


def apply_cad_mesh_frame(xyz: np.ndarray, *, source: Path | None = None) -> np.ndarray:
    """STL/점군에 mesh_rpy 후 mesh_xyz. model.yaml mesh 가 아닐 때는 그대로."""
    pts = _as_xyz(xyz)
    if source is not None:
        try:
            if source.resolve() != cad_mesh_path().resolve():
                return pts
        except FileNotFoundError:
            return pts
    R = cad_mesh_R()
    t = np.asarray(cad_mesh_xyz_mm(), dtype=np.float64).reshape(3)
    if np.allclose(R, np.eye(3)) and np.allclose(t, 0.0):
        return pts
    return (R @ pts.T).T + t


def _as_xyz(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("빈 점군")
    return pts


def _pcd(xyz: np.ndarray):
    o3d = _require_o3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(_as_xyz(xyz))
    return cloud


def _extent_mm(xyz: np.ndarray) -> np.ndarray:
    pts = _as_xyz(xyz)
    return pts.max(axis=0) - pts.min(axis=0)


def _to_mm(xyz: np.ndarray, unit: str, source: Path) -> np.ndarray:
    """yaml 단위 + 크기 보고 mm로 맞춘다. 이 STL은 m로 나온 적이 있다."""
    pts = _as_xyz(xyz)
    span = float(np.max(_extent_mm(pts)))
    u = (unit or "mm").strip().lower()
    if u in {"m", "meter", "meters", "metre", "metres"}:
        print(f"CAD {source.name}: unit={u} → ×1000 mm")
        return pts * 1000.0
    if span < 2.0:
        print(
            f"CAD {source.name}: 크기 {span:.4f} (yaml은 mm). "
            "m로 보고 ×1000"
        )
        return pts * 1000.0
    return pts


def load_ply_xyz(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """roi_cloud.py 가 쓰는 ASCII ply. xyz, 있으면 rgb."""
    text = Path(path).read_text(encoding="ascii").splitlines()
    n = 0
    has_rgb = False
    header_end = 0
    for i, line in enumerate(text):
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
        if line.startswith("property uchar red"):
            has_rgb = True
        if line.strip() == "end_header":
            header_end = i
            break
    rows = []
    for line in text[header_end + 1 :]:
        if not line.strip():
            continue
        rows.append([float(v) for v in line.split()])
    arr = np.asarray(rows, dtype=np.float64)
    if n and len(arr) != n:
        raise ValueError(f"PLY 점 수 불일치: header={n} rows={len(arr)}  {path}")
    if len(arr) == 0:
        raise ValueError(f"빈 PLY: {path}")
    xyz = arr[:, :3].astype(np.float64)
    rgb = arr[:, 3:6].astype(np.uint8) if has_rgb and arr.shape[1] >= 6 else None
    return xyz, rgb


def load_cad_xyz(path: Path | None = None, *, voxel_mm: float = 2.0) -> np.ndarray:
    """CAD ply/mesh → mm 점군. model.yaml mesh 면 mesh_rpy 적용."""
    o3d = _require_o3d()
    source = Path(path) if path is not None else cad_mesh_path()
    if not source.is_file():
        raise FileNotFoundError(f"CAD 없음: {source}")
    unit = cad_unit()
    rpy = cad_mesh_rpy_deg() if source.resolve() == cad_mesh_path().resolve() else (0.0, 0.0, 0.0)
    xyz0 = cad_mesh_xyz_mm() if source.resolve() == cad_mesh_path().resolve() else (0.0, 0.0, 0.0)
    key = (str(source.resolve()), float(voxel_mm), unit, rpy, xyz0)
    cached = _CAD_CACHE.get(key)
    if cached is not None:
        return cached.copy()

    mesh = o3d.io.read_triangle_mesh(str(source))
    if mesh.has_triangles() and len(mesh.triangles) > 0:
        verts = apply_cad_mesh_frame(_to_mm(np.asarray(mesh.vertices), unit, source), source=source)
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.compute_vertex_normals()
        area = float(mesh.get_surface_area())
        n_sample = int(np.clip(area / max(voxel_mm ** 2, 1e-6), 800, 80000))
        try:
            pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
        except RuntimeError:
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
    else:
        pcd = o3d.io.read_point_cloud(str(source))
        if not pcd.has_points():
            raise RuntimeError(f"CAD를 못 읽음: {source}")
        pts = apply_cad_mesh_frame(_to_mm(np.asarray(pcd.points), unit, source), source=source)
        pcd.points = o3d.utility.Vector3dVector(pts)

    if voxel_mm > 0:
        pcd = pcd.voxel_down_sample(voxel_mm)
    xyz = np.asarray(pcd.points, dtype=np.float64)
    if len(xyz) < MIN_POINTS:
        raise RuntimeError(f"CAD 점이 너무 적음: {len(xyz)}  {source}")
    span = _extent_mm(xyz)
    extra = ""
    if any(abs(v) > 1e-12 for v in rpy):
        extra += f"  mesh_rpy={list(rpy)}"
    if any(abs(v) > 1e-12 for v in xyz0):
        extra += f"  mesh_xyz={list(xyz0)}"
    rpy_txt = extra
    print(
        f"CAD {source.name}  n={len(xyz)}  "
        f"span={span[0]:.1f}×{span[1]:.1f}×{span[2]:.1f} mm{rpy_txt}"
    )
    _CAD_CACHE[key] = xyz
    return xyz.copy()


def _preprocess(
    pcd,
    voxel_mm: float,
    *,
    camera_origin: np.ndarray | None,
):
    o3d = _require_o3d()
    cloud = pcd.voxel_down_sample(voxel_mm) if voxel_mm > 0 else pcd
    if len(cloud.points) < MIN_POINTS:
        cloud = pcd
    radius = max(float(voxel_mm) * 2.0, 4.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    if camera_origin is not None:
        cloud.orient_normals_towards_camera_location(
            np.asarray(camera_origin, dtype=np.float64).reshape(3)
        )
    else:
        cloud.orient_normals_consistent_tangent_plane(k=15)
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 2.5, max_nn=100),
    )
    return cloud, fpfh


def _nn_dists(query: np.ndarray, ref: np.ndarray) -> np.ndarray:
    q = _pcd(query)
    r = _pcd(ref)
    return np.asarray(q.compute_point_cloud_distance(r), dtype=np.float64)


def scene_coverage(
    cad_xyz: np.ndarray,
    scene_xyz: np.ndarray,
    T: np.ndarray,
    *,
    thresh_mm: float = 3.0,
) -> dict:
    """회색(scene) 점이 변환된 CAD 위에 얼마나 올라갔는지."""
    aligned = apply_T(_as_xyz(cad_xyz), np.asarray(T, dtype=np.float64))
    dist = _nn_dists(_as_xyz(scene_xyz), aligned)
    return {
        "frac": float(np.mean(dist <= thresh_mm)),
        "median": float(np.median(dist)),
        "p90": float(np.percentile(dist, 90)),
        "max": float(np.max(dist)),
    }


def visible_xyz(
    xyz: np.ndarray,
    camera_origin: np.ndarray | None = None,
) -> np.ndarray:
    """카메라에서 보이는 CAD 점만. 안 보이는 면을 점수·오버레이에서 뺀다."""
    pts = _as_xyz(xyz)
    cam = (
        np.zeros(3, dtype=np.float64)
        if camera_origin is None
        else np.asarray(camera_origin, dtype=np.float64).reshape(3)
    )
    o3d = _require_o3d()
    pcd = _pcd(pts)
    try:
        span = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        radius = max(span * 100.0, 100.0)
        _, idx = pcd.hidden_point_removal(cam, radius)
        if len(idx) >= MIN_POINTS:
            return pts[np.asarray(idx, dtype=np.int64)]
    except Exception:
        pass
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=4.0, max_nn=30)
    )
    nrm = np.asarray(pcd.normals)
    keep = np.sum(nrm * (cam.reshape(1, 3) - pts), axis=1) > 0.0
    if int(np.count_nonzero(keep)) >= MIN_POINTS:
        return pts[keep]
    return pts


def _xy_iou(a: np.ndarray, b: np.ndarray, *, cell_mm: float = 2.0) -> float:
    """탑뷰(XY) 점유 IoU. ㄴ vs 구멍 난 판을 가른다."""
    pa = np.asarray(a, dtype=np.float64).reshape(-1, 3)[:, :2]
    pb = np.asarray(b, dtype=np.float64).reshape(-1, 3)[:, :2]
    if len(pa) == 0 or len(pb) == 0:
        return 0.0
    lo = np.minimum(pa.min(axis=0), pb.min(axis=0))
    cell = max(float(cell_mm), 1.0)

    def cells(p: np.ndarray) -> set[tuple[int, int]]:
        q = np.floor((p - lo) / cell).astype(np.int32)
        return set(map(tuple, q.tolist()))

    ca, cb = cells(pa), cells(pb)
    if not ca or not cb:
        return 0.0
    return float(len(ca & cb) / len(ca | cb))


def pose_coverage(
    cad_xyz: np.ndarray,
    scene_xyz: np.ndarray,
    T: np.ndarray,
    *,
    thresh_mm: float = 3.0,
    camera_origin: np.ndarray | None = None,
    detail: bool = True,
) -> dict:
    """회색→CAD. detail=False 면 가시성·IoU를 건너뛰어 후보 비교만 한다."""
    cad = _as_xyz(cad_xyz)
    scene = _as_xyz(scene_xyz)
    aligned = apply_T(cad, np.asarray(T, dtype=np.float64))
    scene_d = _nn_dists(scene, aligned)
    scene_center = (scene.min(axis=0) + scene.max(axis=0)) * 0.5
    cad_center = (aligned.min(axis=0) + aligned.max(axis=0)) * 0.5
    cov = {
        "frac": float(np.mean(scene_d <= thresh_mm)),
        "vis_frac": 0.0,
        "xy_iou": 0.0,
        "median": float(np.median(scene_d)),
        "p90": float(np.percentile(scene_d, 90)),
        "max": float(np.max(scene_d)),
        "center_delta": float(np.linalg.norm(scene_center - cad_center)),
        "n_vis": int(len(aligned)),
    }
    if detail:
        vis = visible_xyz(aligned, camera_origin)
        vis_d = _nn_dists(vis, scene)
        cov["vis_frac"] = float(np.mean(vis_d <= thresh_mm))
        cov["xy_iou"] = _xy_iou(scene, vis)
        cov["n_vis"] = int(len(vis))
    cov["aligned_ok"] = bool(
        cov["frac"] >= 0.80
        and cov["median"] <= 1.5
        and cov["p90"] <= 4.0
        and cov["center_delta"] <= 6.0
    )
    return cov


def _score_key(cov: dict) -> tuple:
    return (
        float(cov["frac"]),
        -float(cov["median"]),
        -float(cov.get("p90", cov["median"])),
        -float(cov.get("center_delta", 1e9)),
        float(cov.get("vis_frac", 0.0)),
        float(cov.get("xy_iou", 0.0)),
    )


def _icp_from(source, target, T, dist_mm: float, plane: bool, *, iters: int = 80):
    o3d = _require_o3d()
    est = (
        o3d.pipelines.registration.TransformationEstimationPointToPlane()
        if plane
        else o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return o3d.pipelines.registration.registration_icp(
        source,
        target,
        dist_mm,
        np.asarray(T, dtype=np.float64),
        est,
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)),
    )


def _R_axis(axis: int, deg: float) -> np.ndarray:
    a = np.deg2rad(float(deg))
    c, s = float(np.cos(a)), float(np.sin(a))
    if axis == 0:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == 1:
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _yaw_rots() -> list[np.ndarray]:
    """책상 위 ㄴ은 yaw(CAD Z)만 90° 주기. Rx/Ry 는 팔을 세워서 축이 ㄴ 안으로 들어간다."""
    return [_R_axis(2, deg) for deg in (0.0, 90.0, 180.0, 270.0)]


def _shift_centroids(T: np.ndarray, cad: np.ndarray, scene: np.ndarray) -> np.ndarray:
    """CAD 무게중심을 회색 중심에 맞춘다. 코너만 붙은 해를 끌어온다."""
    T2 = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    aligned = apply_T(cad, T2)
    T2[:3, 3] += np.median(scene, axis=0) - np.median(aligned, axis=0)
    return T2


def _icp_multiscale(source, target, T, *, voxel_mm: float) -> np.ndarray:
    Tcur = np.asarray(T, dtype=np.float64).reshape(4, 4)
    v = max(float(voxel_mm), 0.8)
    stages = (
        (max(8.0, v * 6.0), True, 80),
        (max(4.0, v * 3.0), True, 80),
        (max(2.0, v * 1.5), True, 100),
        (max(1.2, v), False, 80),
    )
    for dist, plane, iters in stages:
        result = _icp_from(source, target, Tcur, dist, plane, iters=iters)
        if result.fitness > 0.0:
            Tcur = np.asarray(result.transformation, dtype=np.float64)
    return Tcur


def _apply_cov(best: dict, T: np.ndarray, cov: dict, method: str) -> None:
    best["T"] = T
    best["icp_method"] = method
    best["fitness"] = cov["frac"]
    best["inlier_rmse"] = cov["median"]
    best["scene_frac"] = cov["frac"]
    best["scene_med"] = cov["median"]
    best["scene_p90"] = cov["p90"]
    best["vis_frac"] = cov["vis_frac"]
    best["xy_iou"] = cov["xy_iou"]
    best["aligned_ok"] = cov["aligned_ok"]


def _refine_candidates(
    cad: np.ndarray,
    scene: np.ndarray,
    source,
    target,
    T0: np.ndarray,
    *,
    camera_origin: np.ndarray,
    voxel_mm: float,
    try_90: bool,
) -> tuple[np.ndarray, dict, str]:
    """무게중심 + 다단 ICP. 첫 등록이면 ㄴ 90°도 같이 본다."""
    rots = _yaw_rots() if try_90 else [np.eye(3)]
    T0 = np.asarray(T0, dtype=np.float64).reshape(4, 4)
    best_T = T0
    best_cov = pose_coverage(cad, scene, T0, thresh_mm=3.0, camera_origin=camera_origin)
    best_key = _score_key(best_cov)
    best_method = "icp_hold"
    for Rloc in rots:
        T = T0.copy()
        T[:3, :3] = T0[:3, :3] @ Rloc
        T = _icp_multiscale(source, target, T, voxel_mm=voxel_mm)
        cov = pose_coverage(cad, scene, T, thresh_mm=3.0, camera_origin=camera_origin)
        key = _score_key(cov)
        if key > best_key:
            best_key = key
            best_T = T
            best_cov = cov
            best_method = "icp_refine"
    return best_T, best_cov, best_method


def _ransac_once(source, target, source_fpfh, target_fpfh, dist: float):
    o3d = _require_o3d()
    return o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        False,
        dist,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(dist),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999),
    )


def _clean_registration_scene(scene_xyz: np.ndarray) -> np.ndarray:
    """등록 전에 고립된 뎁스 점만 완만하게 제거한다."""
    scene = _as_xyz(scene_xyz)
    if len(scene) < 40:
        return scene
    cleaned, _ = _pcd(scene).remove_statistical_outlier(
        nb_neighbors=min(20, len(scene) - 1),
        std_ratio=2.0,
    )
    points = np.asarray(cleaned.points, dtype=np.float64)
    return points if len(points) >= MIN_POINTS else scene


def _snap_cad_90(
    cad: np.ndarray,
    scene: np.ndarray,
    source,
    target,
    T_cad_scene: np.ndarray,
    *,
    camera_origin: np.ndarray,
    cover_mm: float,
    up: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    """CAD +Z 를 위로 맞춘 뒤, ㄴ yaw 만 0/90/180/270 으로 고른다."""
    T0 = np.asarray(T_cad_scene, dtype=np.float64).reshape(4, 4)
    if up is not None:
        T0 = maybe_flip_into_table(T0, up_base=up)
    best_T = T0
    best_cov = pose_coverage(
        cad, scene, T0, thresh_mm=cover_mm, camera_origin=camera_origin, detail=False
    )
    best_key = _score_key(best_cov)
    for Rloc in _yaw_rots():
        T = T0.copy()
        T[:3, :3] = T0[:3, :3] @ Rloc
        T_sc = _partial_icp_to_cad(source, target, np.linalg.inv(T))
        T = np.linalg.inv(T_sc)
        if up is not None:
            T = maybe_flip_into_table(T, up_base=up)
        cov = pose_coverage(
            cad, scene, T, thresh_mm=cover_mm, camera_origin=camera_origin, detail=False
        )
        key = _score_key(cov)
        if key > best_key:
            best_key = key
            best_T = T
            best_cov = cov
    return best_T, best_cov


def _partial_icp_to_cad(source, target, T_scene_cad: np.ndarray) -> np.ndarray:
    """부분 장면→전체 CAD 방향으로 coarse-to-fine ICP."""
    T = np.asarray(T_scene_cad, dtype=np.float64).reshape(4, 4)
    stages = (
        (5.0, False, 80),
        (3.0, True, 100),
        (2.0, False, 100),
        (1.2, True, 100),
    )
    for dist, plane, iters in stages:
        result = _icp_from(source, target, T, dist, plane, iters=iters)
        if result.fitness > 0.0:
            T = np.asarray(result.transformation, dtype=np.float64)
    return T


def register_fpfh_icp(
    cad_xyz: np.ndarray,
    scene_xyz: np.ndarray,
    *,
    voxel_mm: float = 2.0,
    tries: int = 8,
    camera_origin: np.ndarray | None = None,
    up: np.ndarray | None = None,
) -> dict:
    """부분 장면→전체 CAD로 등록한 뒤 CAD→장면 T를 반환한다."""
    o3d = _require_o3d()
    cad = _as_xyz(cad_xyz)
    scene = _clean_registration_scene(scene_xyz)
    if len(cad) < MIN_POINTS or len(scene) < MIN_POINTS:
        raise ValueError(f"점 부족  cad={len(cad)} scene={len(scene)}  최소 {MIN_POINTS}")
    cam = np.zeros(3) if camera_origin is None else np.asarray(camera_origin, dtype=np.float64)

    # 장면은 CAD의 일부만 보인다. 전체 CAD→부분 장면 ICP는 안 보이는 CAD
    # 표면까지 억지로 끌어당기므로 반대로 부분 장면→전체 CAD를 푼다.
    source, source_fpfh = _preprocess(_pcd(scene), voxel_mm, camera_origin=cam)
    target, target_fpfh = _preprocess(_pcd(cad), voxel_mm, camera_origin=None)
    v = max(float(voxel_mm), 0.4)
    dist = max(v * 2.5, 2.5)
    cover_mm = 3.0

    best: dict | None = None
    best_key = None
    for i in range(max(1, int(tries))):
        try:
            o3d.utility.random.seed(i + 1)
        except Exception:
            pass
        ransac = _ransac_once(source, target, source_fpfh, target_fpfh, dist)
        if ransac.fitness <= 0.0:
            continue
        T_sc_init = np.asarray(ransac.transformation, dtype=np.float64)
        T_scene_cad = _partial_icp_to_cad(source, target, T_sc_init)
        T = np.linalg.inv(T_scene_cad)
        cov = pose_coverage(
            cad, scene, T, thresh_mm=cover_mm, camera_origin=cam, detail=False
        )
        key = _score_key(cov)
        if best is None or key > best_key:
            best_key = key
            best = {
                "T": T,
                "T_init": np.linalg.inv(T_sc_init),
                "fitness": cov["frac"],
                "inlier_rmse": cov["median"],
                "fitness_ransac": float(ransac.fitness),
                "inlier_rmse_ransac": float(ransac.inlier_rmse),
                "icp_method": "icp_partial",
                "scene_frac": cov["frac"],
                "scene_med": cov["median"],
                "scene_p90": cov["p90"],
                "vis_frac": cov["vis_frac"],
                "xy_iou": cov["xy_iou"],
                "center_delta": cov["center_delta"],
                "aligned_ok": cov["aligned_ok"],
                "n_cad": int(len(target.points)),
                "n_scene": int(len(source.points)),
                "voxel_mm": float(voxel_mm),
                "source": "partial_fpfh_icp",
            }

    if best is None:
        raise RuntimeError(
            "FPFH+RANSAC 실패. ROI 점이 너무 적거나 평면만 남았을 수 있습니다. "
            f"cad={len(target.points)} scene={len(source.points)} voxel={voxel_mm}"
        )

    T_snap, _ = _snap_cad_90(
        cad,
        scene,
        source,
        target,
        best["T"],
        camera_origin=cam,
        cover_mm=cover_mm,
        up=up,
    )
    best["T"] = T_snap
    best["icp_method"] = "icp_partial_90"

    cov = pose_coverage(
        cad, scene, best["T"], thresh_mm=cover_mm, camera_origin=cam, detail=True
    )
    best["fitness"] = cov["frac"]
    best["inlier_rmse"] = cov["median"]
    best["scene_frac"] = cov["frac"]
    best["scene_med"] = cov["median"]
    best["scene_p90"] = cov["p90"]
    best["vis_frac"] = cov["vis_frac"]
    best["xy_iou"] = cov["xy_iou"]
    best["center_delta"] = cov["center_delta"]
    best["aligned_ok"] = cov["aligned_ok"]

    if not best.get("aligned_ok", False):
        print(
            "경고: CAD가 회색 ㄴ과 안 겹침. "
            f"on={best['scene_frac']:.2f} vis={best.get('vis_frac', 0):.2f} "
            f"iou={best.get('xy_iou', 0):.2f}  "
            f"center={best.get('center_delta', 0):.1f} "
            f"med={best['scene_med']:.1f} mm. c 로 다시."
        )
    return best


def T_to_xyzrpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """4x4 → xyz mm, rpy deg. XYZ extrinsic = ZYX intrinsic. R = Rz @ Ry @ Rx."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    xyz = T[:3, 3].copy()
    r00, r10, r20 = float(T[0, 0]), float(T[1, 0]), float(T[2, 0])
    r21, r22 = float(T[2, 1]), float(T[2, 2])
    pitch = np.arcsin(np.clip(-r20, -1.0, 1.0))
    cy = np.cos(pitch)
    if abs(cy) > 1e-8:
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)
    else:
        roll = np.arctan2(-float(T[1, 2]), float(T[1, 1]))
        yaw = 0.0
    rpy = np.degrees(np.array([roll, pitch, yaw], dtype=np.float64))
    return xyz, rpy


def maybe_flip_into_table(T: np.ndarray, *, up_base: np.ndarray | None = None) -> np.ndarray:
    """그리퍼 접근이 책상 아래(-Z)로 꽂히면 CAD Z 를 180° 뒤집는다."""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4).copy()
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64) if up_base is None else np.asarray(
        up_base, dtype=np.float64
    ).reshape(3)
    n = float(np.linalg.norm(up))
    if n < 1e-12:
        return T
    up = up / n
    if float(T[:3, 2] @ up) >= 0.0:
        return T
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        dtype=np.float64,
    )
    return T @ rt_to_T(rx, np.zeros(3))


def refine_icp(
    cad_xyz: np.ndarray,
    scene_xyz: np.ndarray,
    T_init: np.ndarray,
    *,
    voxel_mm: float = 2.0,
    camera_origin: np.ndarray | None = None,
) -> dict:
    """이전 CAD→장면 T에서 부분 장면→CAD ICP로 추적한다."""
    cad = _as_xyz(cad_xyz)
    scene = _clean_registration_scene(scene_xyz)
    if len(cad) < MIN_POINTS or len(scene) < MIN_POINTS:
        raise ValueError(f"점 부족  cad={len(cad)} scene={len(scene)}")
    cam = np.zeros(3) if camera_origin is None else np.asarray(camera_origin, dtype=np.float64)
    source, _ = _preprocess(_pcd(scene), voxel_mm, camera_origin=cam)
    target, _ = _preprocess(_pcd(cad), voxel_mm, camera_origin=None)
    T0 = np.asarray(T_init, dtype=np.float64).reshape(4, 4)
    T_scene_cad = _partial_icp_to_cad(
        source,
        target,
        np.linalg.inv(T0),
    )
    best_T = np.linalg.inv(T_scene_cad)
    cov = pose_coverage(cad, scene, best_T, thresh_mm=3.0, camera_origin=cam)
    return {
        "T": best_T,
        "T_init": T0,
        "fitness": cov["frac"],
        "inlier_rmse": cov["median"],
        "fitness_ransac": 0.0,
        "inlier_rmse_ransac": 0.0,
        "icp_method": "icp_partial_track",
        "scene_frac": cov["frac"],
        "scene_med": cov["median"],
        "scene_p90": cov["p90"],
        "vis_frac": cov["vis_frac"],
        "xy_iou": cov["xy_iou"],
        "center_delta": cov["center_delta"],
        "aligned_ok": cov["aligned_ok"],
        "n_cad": int(len(target.points)),
        "n_scene": int(len(source.points)),
        "voxel_mm": float(voxel_mm),
        "source": "partial_icp_track",
    }


def stabilize_T(
    T_new: np.ndarray,
    T_prev: np.ndarray | None,
    *,
    pivot: np.ndarray | None = None,
) -> np.ndarray:
    """L 브라켓 180° 후보 중 이전 축에 가까운 것을 고른다."""
    Tn = np.asarray(T_new, dtype=np.float64).reshape(4, 4)
    if T_prev is None:
        return Tn
    Tp = np.asarray(T_prev, dtype=np.float64).reshape(4, 4)
    locals_180 = (
        np.eye(3),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    best = Tn
    best_score = -1e18
    Rp = Tp[:3, :3]
    for Rloc in locals_180:
        R = Tn[:3, :3] @ Rloc
        score = float(np.trace(R.T @ Rp))
        if score > best_score:
            best_score = score
            best = Tn.copy()
            best[:3, :3] = R
            if pivot is not None:
                p = np.asarray(pivot, dtype=np.float64).reshape(3)
                world_pivot = Tn[:3, :3] @ p + Tn[:3, 3]
                best[:3, 3] = world_pivot - R @ p
    return best


def register_pose(
    scene_xyz: np.ndarray,
    *,
    cad_path: Path | None = None,
    T_base_cam: np.ndarray | None = None,
    voxel_mm: float = 2.0,
    flip: bool = True,
    tries: int = 8,
    T_init: np.ndarray | None = None,
    T_prev: np.ndarray | None = None,
    camera_origin: np.ndarray | None = None,
) -> dict:
    """한 인스턴스 점군 → T, xyzrpy.

    T_init 이 있으면 RANSAC 없이 ICP만. T_prev 가 있으면 180°를 고정한다.
    """
    cad_xyz = load_cad_xyz(cad_path, voxel_mm=voxel_mm)
    cam = camera_origin
    up = np.array([0.0, 0.0, 1.0]) if flip and T_base_cam is None else None
    if T_init is not None:
        raw = refine_icp(
            cad_xyz, scene_xyz, T_init, voxel_mm=voxel_mm, camera_origin=cam
        )
    else:
        raw = register_fpfh_icp(
            cad_xyz,
            scene_xyz,
            voxel_mm=voxel_mm,
            tries=tries,
            camera_origin=cam,
            up=up,
        )
    T = raw["T"]
    if T_base_cam is not None:
        T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4) @ T
        if flip:
            T = maybe_flip_into_table(T, up_base=np.array([0.0, 0.0, 1.0]))
    elif flip:
        T = maybe_flip_into_table(T, up_base=np.array([0.0, 0.0, 1.0]))
    T = stabilize_T(T, T_prev, pivot=np.median(cad_xyz, axis=0))
    xyz, rpy = T_to_xyzrpy(T)
    return {
        "T": T,
        "xyz_mm": xyz,
        "rpy_deg": rpy,
        "fitness": raw["fitness"],
        "inlier_rmse": raw["inlier_rmse"],
        "fitness_ransac": raw["fitness_ransac"],
        "n_points": int(len(_as_xyz(scene_xyz))),
        "n_cad": raw["n_cad"],
        "source": raw.get("source", "fpfh_icp"),
        "voxel_mm": float(voxel_mm),
        "T_init": raw["T_init"],
        "cad_xyz": cad_xyz,
        "icp_method": raw.get("icp_method", "icp_plane"),
        "scene_frac": raw.get("scene_frac"),
        "scene_med": raw.get("scene_med"),
        "scene_p90": raw.get("scene_p90"),
        "vis_frac": raw.get("vis_frac"),
        "xy_iou": raw.get("xy_iou"),
        "center_delta": raw.get("center_delta"),
        "aligned_ok": bool(raw.get("aligned_ok", False)),
    }


def _overlay_xyz(
    scene_xyz: np.ndarray,
    scene_rgb: np.ndarray | None,
    cad_xyz: np.ndarray,
    T: np.ndarray,
    *,
    camera_origin: np.ndarray | None = None,
    visible_only: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    aligned = apply_T(cad_xyz, T)
    if visible_only:
        aligned = visible_xyz(aligned, camera_origin)
    cad_rgb = np.tile(np.array([[220, 40, 40]], dtype=np.uint8), (len(aligned), 1))
    if scene_rgb is None or len(scene_rgb) != len(scene_xyz):
        scene_rgb = np.tile(np.array([[180, 180, 180]], dtype=np.uint8), (len(scene_xyz), 1))
    xyz = np.vstack([scene_xyz, aligned])
    rgb = np.vstack([scene_rgb, cad_rgb])
    return xyz, rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="CAD vs ROI ply → xyzrpy")
    parser.add_argument(
        "--cad",
        type=Path,
        default=None,
        help="생략하면 yolo/cad/model.yaml 의 mesh",
    )
    parser.add_argument("--scene", type=Path, required=True, help="인스턴스 ply")
    parser.add_argument("--base", action="store_true", help="T_base_cam 으로 베이스 mm")
    parser.add_argument("--voxel-mm", type=float, default=1.0, help="등록용 CAD/장면 다운샘플(mm)")
    parser.add_argument(
        "--overlay-voxel-mm",
        type=float,
        default=0.5,
        help="빨간 CAD 표시 간격(mm). 등록 voxel과 별개",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="등록 1mm + 빨간 CAD 0.5mm (지금 기본값)",
    )
    parser.add_argument("--no-flip", action="store_true", help="베이스에서 CAD Z 뒤집기 끄기")
    parser.add_argument(
        "--overlay",
        type=Path,
        default=None,
        help="정렬된 CAD(빨강)+장면 ply. 생략하면 scene 옆에 _aligned.ply",
    )
    parser.add_argument("--no-overlay", action="store_true")
    parser.add_argument("--view", action="store_true", help="overlay 를 view_cloud.py 로")
    args = parser.parse_args()
    if args.dense:
        args.voxel_mm = min(float(args.voxel_mm), 1.0)
        args.overlay_voxel_mm = min(float(args.overlay_voxel_mm), 0.5)

    scene_path = Path(args.scene)
    if not scene_path.is_file():
        raise SystemExit(f"장면 ply 없음: {scene_path}")
    if scene_path.name.endswith("_slice.ply"):
        print("주의: *_slice.ply 는 윗면만입니다. 등록은 roi_*_camera_mm.ply 를 쓰세요.")

    scene_xyz, scene_rgb = load_ply_xyz(scene_path)
    print(f"scene {scene_path}  n={len(scene_xyz)}")

    T_bc = None
    frame = "camera_mm"
    if args.base:
        from vision.calib import load_T_base_cam

        T_bc, t_path = load_T_base_cam()
        frame = "base_mm"
        print(f"T_base_cam {t_path}")

    pose = register_pose(
        scene_xyz,
        cad_path=args.cad,
        T_base_cam=T_bc,
        voxel_mm=args.voxel_mm,
        flip=not args.no_flip,
    )
    xyz = pose["xyz_mm"]
    rpy = pose["rpy_deg"]
    print(
        f"source={pose['source']}  icp={pose.get('icp_method')}  frame={frame}\n"
        f"scene_on_cad={pose.get('scene_frac', pose['fitness']):.3f}  "
        f"vis={pose.get('vis_frac', 0):.3f}  iou={pose.get('xy_iou', 0):.3f}  "
        f"ok={pose.get('aligned_ok')}  "
        f"med={pose.get('scene_med', pose['inlier_rmse']):.2f} mm  "
        f"p90={pose.get('scene_p90', 0):.2f} mm  "
        f"ransac={pose['fitness_ransac']:.3f}\n"
        f"xyz_mm=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}]\n"
        f"rpy_deg=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]"
    )

    overlay_path = None
    if not args.no_overlay:
        overlay_path = args.overlay or scene_path.with_name(f"{scene_path.stem}_aligned.ply")
        T_overlay = pose["T"]
        cad_for_overlay = load_cad_xyz(args.cad, voxel_mm=float(args.overlay_voxel_mm))
        if T_bc is not None:
            # 장면 ply 는 카메라 mm. 오버레이도 그 프레임에 그린다.
            T_overlay = np.linalg.inv(T_bc) @ pose["T"]
        o_xyz, o_rgb = _overlay_xyz(scene_xyz, scene_rgb, cad_for_overlay, T_overlay)
        write_ply(overlay_path, o_xyz, o_rgb)
        print(f"overlay {overlay_path}  n={len(o_xyz)}  (회색=ROI, 빨강=CAD, c=CAD on/off)")

    if args.view:
        if overlay_path is None:
            raise SystemExit("--view 는 overlay 가 필요합니다. --no-overlay 를 빼세요.")
        view_cmd = [
            sys.executable,
            str(Path(__file__).resolve().parent / "view_cloud.py"),
            str(overlay_path),
            "--no-obb",
            "--cad-T="
            + ",".join(f"{x:.9g}" for x in np.asarray(T_overlay).reshape(-1)),
        ]
        if args.dense or float(args.overlay_voxel_mm) <= 1.0:
            view_cmd.extend(["--point-size", "2"])
        subprocess.run(view_cmd, check=False)


if __name__ == "__main__":
    main()
