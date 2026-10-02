"""cad/model.yaml 과 STL 읽기. 펜던트, 등록, 픽이 같은 값을 쓴다.

PyYAML 없이 이 파일 모양(최상위 키 + 들여쓰기 2칸 섹션)만 읽는다.
길이는 mm, 각도는 deg. 축은 +X 앞, +Y 왼쪽, +Z 위.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np

CAD_DIR = Path(__file__).resolve().parent
CAD_YAML = CAD_DIR / "model.yaml"

# yaml 에 값이 없을 때 쓰는 기본값.
DEFAULT_PLACE_XYZ_MM = (150.0, 150.0, 2.0)
DEFAULT_PLACE_RPY_DEG = (0.0, 0.0, -90.0)
DEFAULT_GRASP_XYZ_MM = (0.0, 0.0, 0.0)
DEFAULT_GRASP_RPY_DEG = (0.0, 0.0, 0.0)
DEFAULT_GRIPPER = 50.0
DEFAULT_DROP_XY_MM = (150.0, -100.0)
DEFAULT_APPROACH_D_MM = 10.0
DEFAULT_APPROACH_A_MM = 30.0  # pre → grasp along axis (mm toward CAD origin)


def read_yaml(path: Path = CAD_YAML) -> dict:
    """최상위 `키: 값` 은 문자열, `섹션:` 아래 2칸 들여쓴 값은 파이썬 리터럴."""
    if not path.is_file():
        return {}
    root: dict = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            section = None
            if value == "":
                section = key
                root[key] = {}
            else:
                root[key] = value.strip("'\"")
            continue
        if section is None or not isinstance(root.get(section), dict):
            continue
        root[section][key] = ast.literal_eval(value) if value else None
    return root


def _vec3(raw: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback
        return _vec3(parsed, fallback)
    return fallback


def _vec2(raw: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 2:
        return (float(raw[0]), float(raw[1]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback
        return _vec2(parsed, fallback)
    return fallback


def _scalar(raw: object, fallback: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _section(data: dict, key: str) -> dict:
    raw = data.get(key)
    return raw if isinstance(raw, dict) else {}


def _top_vec3(key: str, path: Path) -> tuple[float, float, float]:
    """최상위 `key: [a, b, c]`. 없으면 0, 모양이 틀리면 ValueError."""
    raw = read_yaml(path).get(key)
    if not raw:
        return (0.0, 0.0, 0.0)
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{path} {key} 파싱 실패: {raw!r}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(f"{path} {key} 는 숫자 3개여야 함: {raw!r}")
    return (float(parsed[0]), float(parsed[1]), float(parsed[2]))


def mesh_path(path: Path = CAD_YAML) -> Path:
    """yaml 의 mesh. 부품 이름은 yaml 에만 둔다."""
    name = read_yaml(path).get("mesh")
    if not name:
        raise FileNotFoundError(
            f"CAD mesh 없음. {path} 에 `mesh: 파일명` 을 적고 "
            f"파일을 {path.parent}/ 에 두세요."
        )
    mesh = path.parent / name
    if not mesh.is_file():
        raise FileNotFoundError(f"CAD 파일 없음: {mesh}\n{path} 의 mesh 를 확인하세요.")
    return mesh


def unit(path: Path = CAD_YAML) -> str:
    return read_yaml(path).get("unit", "mm") or "mm"


def mesh_rpy_deg(path: Path = CAD_YAML) -> tuple[float, float, float]:
    """STL 파일 → 프로젝트 CAD 프레임. `mesh_rpy: [r,p,y]` (deg)."""
    return _top_vec3("mesh_rpy", path)


def mesh_xyz_mm(path: Path = CAD_YAML) -> tuple[float, float, float]:
    """회전 후 원점 이동(mm). `mesh_xyz: [x,y,z]`."""
    return _top_vec3("mesh_xyz", path)


def class_name(path: Path = CAD_YAML) -> str:
    """부품 이름. 없으면 object."""
    return read_yaml(path).get("class") or "object"


def place_xyzrpy(
    path: Path = CAD_YAML,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """`place` → (xyz_mm, rpy_deg). 화면에 물체를 올려 두는 기본 자세."""
    raw = _section(read_yaml(path), "place")
    return (
        _vec3(raw.get("xyz_mm"), DEFAULT_PLACE_XYZ_MM),
        _vec3(raw.get("rpy_deg"), DEFAULT_PLACE_RPY_DEG),
    )


def grasp_xyzrpy_gripper(
    path: Path = CAD_YAML,
) -> tuple[tuple[float, float, float], tuple[float, float, float], float]:
    """`grasp` → (xyz_mm, rpy_deg, gripper). 물체 원점 기준 집는 점."""
    raw = _section(read_yaml(path), "grasp")
    return (
        _vec3(raw.get("xyz_mm"), DEFAULT_GRASP_XYZ_MM),
        _vec3(raw.get("rpy_deg"), DEFAULT_GRASP_RPY_DEG),
        _scalar(raw.get("gripper"), DEFAULT_GRIPPER),
    )


def drop_xy_mm(path: Path = CAD_YAML) -> tuple[float, float]:
    raw = _section(read_yaml(path), "drop")
    return _vec2(raw.get("xy_mm"), DEFAULT_DROP_XY_MM)


def approach_mm(path: Path = CAD_YAML) -> tuple[float, float]:
    """(d, a). d: 표면 → 대기, a: 대기 → 집기."""
    data = read_yaml(path)
    return (
        _scalar(data.get("approach_d_mm"), DEFAULT_APPROACH_D_MM),
        _scalar(data.get("approach_a_mm"), DEFAULT_APPROACH_A_MM),
    )


def _extent_mm(xyz: np.ndarray) -> np.ndarray:
    return np.ptp(xyz, axis=0)


def as_xyz(xyz: np.ndarray) -> np.ndarray:
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    if len(pts) == 0:
        raise ValueError("빈 점군")
    return pts


def to_mm(xyz: np.ndarray, unit_name: str, source: Path) -> np.ndarray:
    """yaml 단위 + 크기 보고 mm로 맞춘다. 이 STL은 m로 나온 적이 있다."""
    pts = as_xyz(xyz)
    span = float(np.max(_extent_mm(pts)))
    u = (unit_name or "mm").strip().lower()
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


def mesh_R(path: Path = CAD_YAML) -> np.ndarray:
    """mesh_rpy → 3x3. 파일 좌표 → 프로젝트 CAD 프레임. R = Rz @ Ry @ Rx."""
    r, p, y = np.deg2rad(mesh_rpy_deg(path))
    if abs(r) < 1e-12 and abs(p) < 1e-12 and abs(y) < 1e-12:
        return np.eye(3)
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return Rz @ Ry @ Rx


def apply_mesh_frame(xyz: np.ndarray, *, source: Path | None = None) -> np.ndarray:
    """STL/점군에 mesh_rpy 후 mesh_xyz. yaml 의 mesh 가 아닐 때는 그대로."""
    pts = as_xyz(xyz)
    if source is not None:
        try:
            if source.resolve() != mesh_path().resolve():
                return pts
        except FileNotFoundError:
            return pts
    R = mesh_R()
    t = np.asarray(mesh_xyz_mm(), dtype=np.float64).reshape(3)
    if np.allclose(R, np.eye(3)) and np.allclose(t, 0.0):
        return pts
    return (R @ pts.T).T + t


def load_mesh_m(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """STL → (정점 m, 삼각형). CAD 프레임(mesh_rpy·mesh_xyz 적용 후)."""
    import open3d as o3d

    source = Path(path) if path is not None else mesh_path()
    mesh = o3d.io.read_triangle_mesh(str(source))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise RuntimeError(f"삼각형 없음: {source}")
    verts_mm = apply_mesh_frame(to_mm(np.asarray(mesh.vertices), unit(), source), source=source)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    return verts_mm / 1000.0, faces


def bounds_mm(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """CAD 프레임 AABB (mins, maxs) mm."""
    verts_m, _faces = load_mesh_m(path)
    verts_mm = np.asarray(verts_m, dtype=float).reshape(-1, 3) * 1000.0
    return verts_mm.min(axis=0), verts_mm.max(axis=0)
