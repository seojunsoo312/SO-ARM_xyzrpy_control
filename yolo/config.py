"""Single-class YOLO paths and names.

클래스 이름은 여기만 바꾼다. 라벨 파일은 인덱스 0이라 학습 전에 이름을
바꿔도 된다.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CLASS_ID = 0

RAW_IMAGES = ROOT / "datasets" / "raw" / "images"
RAW_LABELS = ROOT / "datasets" / "raw" / "labels"
RAW_LABELS_SEG = ROOT / "datasets" / "raw" / "labels_seg"
SPLITS_DIR = ROOT / "datasets" / "splits"
TRAIN_TXT = SPLITS_DIR / "train.txt"
VAL_TXT = SPLITS_DIR / "val.txt"
SEG_VIEW = ROOT / "datasets" / "seg"
DATA_YAML = ROOT / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
BEST_PT = WEIGHTS_DIR / "best.pt"
BEST_SEG_PT = WEIGHTS_DIR / "best-seg.pt"
_SAM_LOCAL = WEIGHTS_DIR / "mobile_sam.pt"
SAM_MODEL = str(_SAM_LOCAL) if _SAM_LOCAL.is_file() else "mobile_sam.pt"  # Orin 8GB. sam_b.pt 는 무겁다.

CAD_DIR = ROOT / "cad"
CAD_YAML = CAD_DIR / "model.yaml"


def _plain_yaml(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    for raw in path.read_text(encoding="utf-8").splitlines():
        stripped = raw.split("#", 1)[0]
        if stripped[:1] in {" ", "\t"}:
            continue
        line = stripped.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = value.strip().strip("'\"")
    return out


def cad_mesh_path() -> Path:
    """yolo/cad/model.yaml 의 mesh. 부품 이름은 yaml 에만 둔다."""
    spec = _plain_yaml(CAD_YAML)
    name = spec.get("mesh")
    if not name:
        raise FileNotFoundError(
            f"CAD mesh 없음. {CAD_YAML} 에 `mesh: 파일명` 을 적고 "
            f"파일을 {CAD_DIR}/ 에 두세요."
        )
    path = CAD_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"CAD 파일 없음: {path}\n{CAD_YAML} 의 mesh 를 확인하세요.")
    return path


def cad_unit() -> str:
    return _plain_yaml(CAD_YAML).get("unit", "mm") or "mm"


def _yaml_vec3(key: str, *, default: tuple[float, float, float]) -> tuple[float, float, float]:
    import ast

    raw = _plain_yaml(CAD_YAML).get(key)
    if not raw:
        return default
    try:
        parsed = ast.literal_eval(raw)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{CAD_YAML} {key} 파싱 실패: {raw!r}") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 3:
        raise ValueError(f"{CAD_YAML} {key} 는 숫자 3개여야 함: {raw!r}")
    return (float(parsed[0]), float(parsed[1]), float(parsed[2]))


def cad_mesh_rpy_deg() -> tuple[float, float, float]:
    """STL 파일 → 프로젝트 CAD 프레임. model.yaml `mesh_rpy: [r,p,y]` (deg)."""
    return _yaml_vec3("mesh_rpy", default=(0.0, 0.0, 0.0))


def cad_mesh_xyz_mm() -> tuple[float, float, float]:
    """회전 후 원점 이동(mm). model.yaml `mesh_xyz: [x,y,z]`."""
    return _yaml_vec3("mesh_xyz", default=(0.0, 0.0, 0.0))


def class_name() -> str:
    """YOLO 1클래스 이름. cad/model.yaml 의 class, 없으면 object."""
    return _plain_yaml(CAD_YAML).get("class") or "object"


CLASS_NAME = class_name()


def add_class_argument(parser) -> None:
    parser.add_argument(
        "--class-name",
        default=None,
        metavar="NAME",
        help=f"YOLO 1클래스 표시 이름. 생략하면 cad/model.yaml ({CLASS_NAME})",
    )


def class_from_args(args) -> str:
    raw = getattr(args, "class_name", None)
    if raw is None:
        return CLASS_NAME
    name = str(raw).strip()
    return name or CLASS_NAME

VAL_RATIO = 0.2
SPLIT_SEED = 42

# Orin Nano 8GB. 학습이 OOM 나면 batch를 2로.
TRAIN_IMGSZ = 640
TRAIN_BATCH = 4
TRAIN_EPOCHS = 50
TRAIN_WORKERS = 0  # Jetson 공유메모리. PC면 2~4
TRAIN_SEG_BATCH = 2  # seg는 detect보다 VRAM을 더 쓴다
DETECT_CONF = 0.5


def default_start_pt(*, seg: bool) -> str:
    """전이학습 시작 가중치. weights/ 에 있으면 그 경로, 없으면 파일명만 (ultralytics 가 받음)."""
    name = "yolo11n-seg.pt" if seg else "yolo11n-obb.pt"
    local = WEIGHTS_DIR / name
    return str(local) if local.is_file() else name


def quiet_gtk() -> None:
    for key in ("GTK_MODULES", "GTK3_MODULES"):
        raw = os.environ.get(key)
        if not raw:
            continue
        os.environ[key] = ":".join(
            p for p in raw.split(":") if p and "canberra" not in p.lower()
        )
    # conda opencv-python 5 는 Qt GUI. 폰트 경로가 없으면 창이 안 뜨거나 경고만 난다.
    for fonts in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype"),
    ):
        if fonts.is_dir():
            os.environ.setdefault("QT_QPA_FONTDIR", str(fonts))
            break


def ensure_project_on_path() -> None:
    import sys

    path = str(PROJECT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
