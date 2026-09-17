#!/usr/bin/env python3
"""저장한 PLY를 Open3D로 본다.

  python yolo/pose/view/view_cloud.py                 # runs/roi의 최신 PLY
  python yolo/pose/view/view_cloud.py path/to/roi.ply
  c = 빨간 CAD 점군 on/off (overlay ply)
  축: 원점(로봇/작업 프레임) + --cad-T 있으면 CAD.
  회색 격자: 작업 프레임 z=0. 노랑: --desk-plane (책상 RANSAC).
  --no-grid 로 z=0 격자 끔.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from yolo.config import RUNS_DIR

ROI_DIR = RUNS_DIR / "roi"


def _split_cad(pcd):
    """register overlay 의 빨강 CAD 와 나머지(회색 ROI)를 나눈다."""
    if not pcd.has_colors() or not pcd.has_points():
        return pcd, None
    colors = np.asarray(pcd.colors)
    cad = (colors[:, 0] > 0.70) & (colors[:, 1] < 0.30) & (colors[:, 2] < 0.30)
    n_cad = int(np.count_nonzero(cad))
    n_scene = int(len(colors) - n_cad)
    if n_cad == 0 or n_scene == 0:
        return pcd, None
    idx_cad = np.flatnonzero(cad)
    idx_scene = np.flatnonzero(~cad)
    scene = pcd.select_by_index(idx_scene.tolist())
    cad_pcd = pcd.select_by_index(idx_cad.tolist())
    return scene, cad_pcd


def _remove_statistical(pcd, *, nb_neighbors: int, std_ratio: float):
    """멀리 떨어진 회색 점 제거. OBB가 늘어지는 주원인."""
    n = len(pcd.points)
    if n < max(int(nb_neighbors) + 5, 16) or std_ratio <= 0:
        return pcd
    cleaned, _ = pcd.remove_statistical_outlier(
        nb_neighbors=int(nb_neighbors),
        std_ratio=float(std_ratio),
    )
    n_keep = len(cleaned.points)
    n_drop = n - n_keep
    if n_drop > 0:
        print(f"outlier 제거 {n_drop}/{n}  (nb={nb_neighbors} std={std_ratio:g})")
    if n_keep < 8:
        return pcd
    return cleaned


def _clip_far_from_median(pcd, *, k: float = 2.5, min_mm: float = 50.0):
    """본체 중심에서 먼 잔여 점 제거. statistical이 작은 덩어리를 못 지울 때."""
    n = len(pcd.points)
    if n < 8:
        return pcd
    pts = np.asarray(pcd.points)
    center = np.median(pts, axis=0)
    dist = np.linalg.norm(pts - center, axis=1)
    lim = max(float(min_mm), float(np.percentile(dist, 90)) * float(k))
    keep = np.flatnonzero(dist <= lim)
    n_drop = n - int(len(keep))
    if n_drop <= 0:
        return pcd
    print(f"OBB 원거리 제거 {n_drop}/{n}  (>{lim:.0f} mm from median)")
    return pcd.select_by_index(keep.tolist())


def _clean_scene(pcd, *, nb_neighbors: int, std_ratio: float):
    if std_ratio <= 0 or not pcd.has_points():
        return pcd
    n = len(pcd.points)
    if n >= 24:
        try:
            rad, _ = pcd.remove_radius_outlier(nb_points=8, radius=8.0)
            if len(rad.points) >= 8:
                pcd = rad
        except RuntimeError:
            pass
    pcd = _remove_statistical(pcd, nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return _clip_far_from_median(pcd)


def _grid_on_uv(
    origin: np.ndarray,
    u_axis: np.ndarray,
    v_axis: np.ndarray,
    *,
    half_mm: float,
    step_mm: float,
    color,
):
    import open3d as o3d

    half = float(max(half_mm, step_mm))
    step = float(max(step_mm, 1.0))
    us = np.arange(-half, half + 0.5 * step, step)
    vs = np.arange(-half, half + 0.5 * step, step)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    u_axis = np.asarray(u_axis, dtype=np.float64).reshape(3)
    v_axis = np.asarray(v_axis, dtype=np.float64).reshape(3)
    points: list[list[float]] = []
    lines: list[list[int]] = []
    for v in vs:
        i0 = len(points)
        points.append((origin - half * u_axis + v * v_axis).tolist())
        points.append((origin + half * u_axis + v * v_axis).tolist())
        lines.append([i0, i0 + 1])
    for u in us:
        i0 = len(points)
        points.append((origin + u * u_axis - half * v_axis).tolist())
        points.append((origin + u * u_axis + half * v_axis).tolist())
        lines.append([i0, i0 + 1])
    grid = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64)),
        lines=o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32)),
    )
    grid.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    )
    return grid


def _xy_grid(
    *,
    half_mm: float,
    step_mm: float,
    z: float = 0.0,
    color=(0.35, 0.35, 0.38),
):
    """Working-frame XY plane (z=constant). Same origin as the axis triad."""
    return _grid_on_uv(
        np.array([0.0, 0.0, z]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        half_mm=half_mm,
        step_mm=step_mm,
        color=color,
    )


def _desk_grid(
    plane: np.ndarray,
    *,
    center: np.ndarray,
    half_mm: float,
    step_mm: float,
    color=(0.95, 0.75, 0.15),
):
    """ax+by+cz+d=0 in the same frame as the point cloud."""
    n = np.asarray(plane[:3], dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(n))
    if length < 1e-12:
        raise ValueError("desk plane normal is zero")
    n = n / length
    d = float(plane[3]) / length
    center = np.asarray(center, dtype=np.float64).reshape(3)
    origin = center - (float(n @ center) + d) * n
    if abs(float(n[2])) < 0.9:
        u = np.cross(n, np.array([0.0, 0.0, 1.0]))
    else:
        u = np.cross(n, np.array([0.0, 1.0, 0.0]))
    u = u / max(float(np.linalg.norm(u)), 1e-12)
    v = np.cross(n, u)
    v = v / max(float(np.linalg.norm(v)), 1e-12)
    return _grid_on_uv(
        origin, u, v, half_mm=half_mm, step_mm=step_mm, color=color
    )


def _geometries(
    pcd,
    show_obb: bool = True,
    show_grid: bool = True,
    T_cad=None,
    desk_plane=None,
):
    import open3d as o3d

    extras = []
    if not pcd.has_points():
        return extras
    bounds = pcd.get_axis_aligned_bounding_box()
    extent = np.asarray(bounds.get_extent(), dtype=float)
    diagonal = float(np.linalg.norm(extent))
    extras.append(
        o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=max(min(diagonal * 0.25, 60.0), 25.0)
        )
    )
    if T_cad is not None:
        cad_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=30.0)
        cad_frame.transform(np.asarray(T_cad, dtype=np.float64).reshape(4, 4))
        extras.append(cad_frame)
    center = np.asarray(bounds.get_center(), dtype=float)
    reach = float(np.linalg.norm(center[:2])) + 0.5 * float(np.linalg.norm(extent[:2]))
    half = max(0.5 * diagonal, reach, 80.0)
    step = max(round(half / 8.0 / 5.0) * 5.0, 10.0)  # ~8 divisions, 5 mm snap
    if show_grid:
        # Cover origin triad + reach toward the cloud (camera clouds sit far on +Z).
        extras.append(_xy_grid(half_mm=half, step_mm=step, z=0.0))
    if desk_plane is not None:
        extras.append(
            _desk_grid(desk_plane, center=center, half_mm=half, step_mm=step)
        )
    if show_obb and len(pcd.points) >= 4:
        try:
            obb = pcd.get_oriented_bounding_box(robust=True)
            obb.color = (0.0, 1.0, 0.0)
            extras.append(obb)
        except RuntimeError as exc:
            print(f"OBB 표시 생략: {exc}")
    return extras


def _show(
    pcd,
    *,
    title: str,
    point_size: float,
    show_obb: bool,
    show_grid: bool,
    outlier_nb: int,
    outlier_std: float,
    T_cad=None,
    desk_plane=None,
) -> None:
    import open3d as o3d

    scene, cad = _split_cad(pcd)
    if outlier_std > 0:
        scene = _clean_scene(
            scene, nb_neighbors=outlier_nb, std_ratio=outlier_std
        )
    vis = o3d.visualization.VisualizerWithKeyCallback()
    if not vis.create_window(window_name=title, width=1000, height=720):
        raise RuntimeError("Open3D 창을 만들 수 없습니다. 데스크톱 세션에서 실행하세요.")
    try:
        vis.add_geometry(scene)
        cad_on = True
        if cad is not None:
            vis.add_geometry(cad)
            print("c = 빨간 CAD on/off")

            def _toggle(_vis):
                nonlocal cad_on
                cad_on = not cad_on
                if cad_on:
                    _vis.add_geometry(cad, reset_bounding_box=False)
                    print("CAD on")
                else:
                    _vis.remove_geometry(cad, reset_bounding_box=False)
                    print("CAD off")
                return False

            vis.register_key_callback(ord("C"), _toggle)
            vis.register_key_callback(ord("c"), _toggle)
        for geometry in _geometries(
            scene,
            show_obb=show_obb,
            show_grid=show_grid,
            T_cad=T_cad,
            desk_plane=desk_plane,
        ):
            vis.add_geometry(geometry)
        options = vis.get_render_option()
        options.point_size = point_size
        options.background_color = np.array([0.04, 0.04, 0.04])
        options.show_coordinate_frame = False
        vis.reset_view_point(True)
        vis.run()
    finally:
        vis.destroy_window()


def _latest_ply() -> Path:
    files = sorted(ROI_DIR.glob("*.ply"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"PLY 없음: {ROI_DIR} (roi_cloud.py에서 s를 누르세요.)")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="PLY point-cloud viewer")
    parser.add_argument("ply", type=Path, nargs="?", default=None)
    parser.add_argument("--point-size", type=float, default=4.0)
    parser.add_argument("--no-obb", action="store_true")
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="XY 평면(z=0) 그리드 끄기",
    )
    parser.add_argument(
        "--no-outlier",
        action="store_true",
        help="회색 점 statistical outlier 제거 끄기",
    )
    parser.add_argument(
        "--outlier-std",
        type=float,
        default=1.5,
        help="작을수록 더 많이 지움. 0이면 끄기",
    )
    parser.add_argument("--outlier-nb", type=int, default=20)
    parser.add_argument(
        "--cad-T",
        default=None,
        help="CAD 4x4 (16숫자). 있으면 CAD 축을 추가로 그림. 원점 축·z=0 격자는 유지",
    )
    parser.add_argument(
        "--desk-plane",
        default=None,
        help="책상 ax,by,cz,d (점군과 같은 프레임). 노란 격자. z=0 회색과 별개",
    )
    args = parser.parse_args()

    path = args.ply or _latest_ply()
    if not path.is_file():
        raise SystemExit(f"파일 없음: {path}")

    try:
        import open3d as o3d
    except ImportError as exc:
        raise SystemExit(
            "open3d가 없습니다. 지금 환경에 설치하세요:\n"
            "  pip install open3d"
        ) from exc

    T_cad = None
    if args.cad_T:
        vals = [float(x) for x in str(args.cad_T).replace(",", " ").split()]
        if len(vals) != 16:
            raise SystemExit(f"--cad-T 는 16개 숫자여야 합니다. 지금 {len(vals)}개")
        T_cad = np.asarray(vals, dtype=np.float64).reshape(4, 4)

    desk_plane = None
    if args.desk_plane:
        vals = [float(x) for x in str(args.desk_plane).replace(",", " ").split()]
        if len(vals) != 4:
            raise SystemExit(f"--desk-plane 은 a,b,c,d 4개여야 합니다. 지금 {len(vals)}개")
        desk_plane = np.asarray(vals, dtype=np.float64)
        n = desk_plane[:3]
        nz = float(n[2])
        if abs(nz) > 1e-6:
            z0 = -float(desk_plane[3]) / nz
            print(f"책상 평면  x=y=0에서 z={z0:.2f} mm  (노랑). 회색 격자는 로봇 z=0")
        else:
            print("책상 평면 표시 (노랑). 회색 격자는 로봇 z=0")

    pcd = o3d.io.read_point_cloud(str(path))
    print(path, "points", len(pcd.points))
    _show(
        pcd,
        title=str(path),
        point_size=args.point_size,
        show_obb=not args.no_obb,
        show_grid=not args.no_grid,
        outlier_nb=args.outlier_nb,
        outlier_std=0.0 if args.no_outlier else float(args.outlier_std),
        T_cad=T_cad,
        desk_plane=desk_plane,
    )


if __name__ == "__main__":
    main()
