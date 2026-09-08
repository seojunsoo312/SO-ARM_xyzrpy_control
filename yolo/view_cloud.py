#!/usr/bin/env python3
"""저장한 PLY를 Open3D로 본다.

  python yolo/view_cloud.py                 # runs/roi의 최신 PLY
  python yolo/view_cloud.py path/to/roi.ply
  c = 빨간 CAD 점군 on/off (overlay ply)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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


def _geometries(pcd, show_obb: bool = True):
    import open3d as o3d

    extras = []
    if not pcd.has_points():
        return extras
    bounds = pcd.get_axis_aligned_bounding_box()
    diagonal = float(np.linalg.norm(bounds.get_extent()))
    axis_size = max(diagonal * 0.35, 5.0)
    extras.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size))
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
    outlier_nb: int,
    outlier_std: float,
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
        for geometry in _geometries(scene, show_obb=show_obb):
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

    pcd = o3d.io.read_point_cloud(str(path))
    print(path, "points", len(pcd.points))
    _show(
        pcd,
        title=str(path),
        point_size=args.point_size,
        show_obb=not args.no_obb,
        outlier_nb=args.outlier_nb,
        outlier_std=0.0 if args.no_outlier else float(args.outlier_std),
    )


if __name__ == "__main__":
    main()
