"""YOLO paths and pose-class names.

자세 클래스는 POSE_CLASSES 가 정본이다. 라벨 txt 의 맨 앞 숫자가 그 인덱스다.
기존 파일의 0 은 지우지 않는다. 라벨 UI에서 눌러 고치기 전까지 세우기로 읽힌다.
"""

from __future__ import annotations

import os
from pathlib import Path

from cad.model import class_name

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

CLASS_ID = 0
# (인덱스, data.yaml 이름, 라벨 화면). 베이스 자세: 세우기 / 눕히기 / 비스듬히.
POSE_CLASSES: tuple[tuple[int, str, str], ...] = (
    (0, "세우기", "세우기"),
    (1, "눕히기", "눕히기"),
    (2, "비스듬히", "비스듬히"),
)

RAW_IMAGES = ROOT / "datasets" / "raw" / "images"
RAW_LABELS = ROOT / "datasets" / "raw" / "labels"
SPLITS_DIR = ROOT / "datasets" / "splits"
TRAIN_TXT = SPLITS_DIR / "train.txt"
VAL_TXT = SPLITS_DIR / "val.txt"
DATA_YAML = ROOT / "data.yaml"
WEIGHTS_DIR = ROOT / "weights"
RUNS_DIR = ROOT / "runs"
# 등록 요청 파일과 대기 시간은 yolo/pose/register_link.py.
BEST_PT = WEIGHTS_DIR / "best.pt"

# 부품 이름(class)은 cad/model.yaml 에만 둔다. 읽기는 cad.model.
CLASS_NAME = class_name()


def pose_class_ids() -> tuple[int, ...]:
    return tuple(cid for cid, _name, _ko in POSE_CLASSES)


def pose_name(class_id: int, *, korean: bool = False) -> str:
    for cid, name, ko in POSE_CLASSES:
        if cid == int(class_id):
            return ko if korean else name
    return str(class_id)


def pose_names_yaml() -> str:
    lines = [f"nc: {len(POSE_CLASSES)}", "names:"]
    for cid, name, _ko in POSE_CLASSES:
        lines.append(f"  {cid}: {name}")
    return "\n".join(lines) + "\n"


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
DETECT_CONF = 0.5


def default_start_pt() -> str:
    """전이학습 시작 가중치. weights/ 에 있으면 그 경로, 없으면 파일명만 (ultralytics 가 받음)."""
    name = "yolo11n.pt"
    local = WEIGHTS_DIR / name
    return str(local) if local.is_file() else name


def _system_font_dir() -> Path | None:
    for fonts in (
        Path("/usr/share/fonts/truetype/dejavu"),
        Path("/usr/share/fonts/truetype/liberation"),
        Path("/usr/share/fonts/truetype"),
    ):
        if fonts.is_dir() and any(fonts.glob("*.ttf")):
            return fonts
    return None


def _link_cv2_qt_fonts(fonts: Path) -> None:
    """opencv-python 의 Qt 는 fontconfig 가 없고 cv2/qt/fonts 만 본다."""
    import sys

    ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    qt_dirs = [
        Path(sys.executable).resolve().parent.parent / "lib" / ver / "site-packages" / "cv2" / "qt",
    ]
    for entry in sys.path:
        qt_dirs.append(Path(entry) / "cv2" / "qt")
    target = fonts.resolve()
    for qt in qt_dirs:
        if not qt.is_dir():
            continue
        dest = qt / "fonts"
        try:
            if dest.is_symlink() and dest.resolve() == target:
                return
            if dest.exists():
                return
            dest.symlink_to(target, target_is_directory=True)
        except OSError:
            continue
        return


def quiet_gtk() -> None:
    for key in ("GTK_MODULES", "GTK3_MODULES"):
        raw = os.environ.get(key)
        if not raw:
            continue
        os.environ[key] = ":".join(
            p for p in raw.split(":") if p and "canberra" not in p.lower()
        )
    # conda opencv-python 의 Qt 는 글꼴을 안 넣고, cv2/qt/fonts 가 없으면 경고만 낸다.
    fonts = _system_font_dir()
    if fonts is None:
        return
    os.environ["QT_QPA_FONTDIR"] = str(fonts)
    _link_cv2_qt_fonts(fonts)


def ensure_project_on_path() -> None:
    import sys

    path = str(PROJECT_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)
