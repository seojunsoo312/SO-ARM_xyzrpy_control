#!/usr/bin/env python3
"""yolo/cad/ STL 의 파일 축(XYZ)을 그대로 보여 준다.

등록·카메라 없음. 이 축이 roi_cloud.py / register.py 가 맞추는 정본이다.
파일 원점 = (0,0,0), 회전 0. 빨강=X 초록=Y 파랑=Z.

  python yolo/view_cad.py
  python yolo/view_cad.py bracket_4035.stl
  python yolo/view_cad.py /path/to/other.stl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.config import CAD_DIR, cad_mesh_path, cad_unit
from yolo.register import _to_mm


def _resolve_mesh(name: str | None) -> Path:
    if name is None:
        return cad_mesh_path()
    raw = Path(name)
    if raw.is_file():
        return raw.resolve()
    cand = CAD_DIR / raw.name
    if cand.is_file():
        return cand.resolve()
    if raw.suffix == "":
        for ext in (".stl", ".STL", ".ply", ".PLY"):
            hit = CAD_DIR / f"{raw.name}{ext}"
            if hit.is_file():
                return hit.resolve()
    listed = sorted(
        p.name
        for p in CAD_DIR.iterdir()
        if p.suffix.lower() in {".stl", ".ply"} and p.is_file()
    )
    hint = f"  cad/ 에 있는 파일: {', '.join(listed)}" if listed else f"  {CAD_DIR} 에 stl이 없습니다."
    raise FileNotFoundError(f"STL 없음: {name}\n{hint}")


def _load_mesh_mm(path: Path):
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise RuntimeError(f"삼각형 없음: {path}")
    verts = _to_mm(np.asarray(mesh.vertices), cad_unit(), path)
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        mesh.paint_uniform_color((0.72, 0.72, 0.74))
    return mesh


def _print_axes(path: Path, verts: np.ndarray, axis_mm: float) -> None:
    lo = verts.min(axis=0)
    hi = verts.max(axis=0)
    span = hi - lo
    inside = bool(np.all(lo <= 0.0) and np.all(hi >= 0.0))
    print(f"파일  {path}")
    print("CAD 축 = STL 파일 좌표. 원점 (0,0,0) mm, rpy = 0,0,0")
    print("빨강=X  초록=Y  파랑=Z")
    print(
        f"bbox mm  min=[{lo[0]:.1f}, {lo[1]:.1f}, {lo[2]:.1f}]  "
        f"max=[{hi[0]:.1f}, {hi[1]:.1f}, {hi[2]:.1f}]"
    )
    print(f"span mm  {span[0]:.1f} × {span[1]:.1f} × {span[2]:.1f}  축 길이={axis_mm:.0f} mm")
    print(
        "원점이 AABB "
        + ("안 (형상에 원점이 있음)" if inside else "밖 (원점이 형상 밖에 있음)")
    )
    print("roi_cloud.py 의 축이 이 방향과 같아야 등록이 맞다.")


def _show(mesh, *, title: str, axis_mm: float) -> None:
    import open3d as o3d

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=float(axis_mm))
    vis = o3d.visualization.Visualizer()
    if not vis.create_window(window_name=title, width=1000, height=720):
        raise RuntimeError("Open3D 창을 만들 수 없습니다. 데스크톱 세션에서 실행하세요.")
    try:
        vis.add_geometry(mesh)
        vis.add_geometry(axes)
        options = vis.get_render_option()
        options.background_color = np.array([0.04, 0.04, 0.04])
        options.mesh_show_back_face = True
        options.show_coordinate_frame = False
        vis.reset_view_point(True)
        vis.run()
    finally:
        vis.destroy_window()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="yolo/cad STL 파일 축(XYZ) 보기. 등록 없음."
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        default=None,
        help="파일명 또는 경로. 생략하면 model.yaml 의 mesh",
    )
    parser.add_argument(
        "--axis-mm",
        type=float,
        default=30.0,
        help="축 길이(mm). roi_cloud.py 오버레이와 같게 30",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="yolo/cad/ 의 stl/ply 만 나열",
    )
    args = parser.parse_args()

    if args.list:
        files = sorted(
            p.name
            for p in CAD_DIR.iterdir()
            if p.suffix.lower() in {".stl", ".ply"} and p.is_file()
        )
        print(CAD_DIR)
        if not files:
            raise SystemExit("stl/ply 없음")
        for name in files:
            mark = "  (model.yaml)" if name == cad_mesh_path().name else ""
            print(f"  {name}{mark}")
        return

    try:
        import open3d as o3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "open3d가 없습니다. conda activate lerobot 한 뒤 설치하세요."
        ) from exc

    path = _resolve_mesh(args.mesh)
    mesh = _load_mesh_mm(path)
    verts = np.asarray(mesh.vertices)
    _print_axes(path, verts, float(args.axis_mm))
    _show(mesh, title=f"CAD 축  {path.name}", axis_mm=float(args.axis_mm))


if __name__ == "__main__":
    main()
