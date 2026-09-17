#!/usr/bin/env python3
"""교육 산출물을 지워 다음 기수를 처음부터 돌리게 한다.

목록을 보여 준 뒤 y 를 입력해야 지운다.

  python pendant/reset_class.py --yolo          # 촬영·라벨·학습
  python pendant/reset_class.py --calib         # intrinsic + 손눈 샘플·결과
  python pendant/reset_class.py --ply           # 점군 ply
  python pendant/reset_class.py --all           # 위 전부
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PENDANT = Path(__file__).resolve().parent
PROJECT = PENDANT.parent

YOLO = PROJECT / "yolo"
VISION = PROJECT / "vision"

SKIP_DIR_NAMES = {".git", ".venv", "venv", "node_modules"}

def _restore_data_yaml() -> None:
    dest = YOLO / "data.yaml"
    root = YOLO.resolve()
    dest.write_text(
        f"path: {root.as_posix()}\n"
        "train: datasets/splits/train.txt\n"
        "val: datasets/splits/val.txt\n"
        "nc: 1\n"
        "names:\n"
        "  0: bracket\n",
        encoding="utf-8",
    )


def _collect_files(folder: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not folder.is_dir():
        return []
    out: list[Path] = []
    for pat in patterns:
        out.extend(p for p in folder.glob(pat) if p.is_file())
    return sorted(set(out))


def _collect_tree(path: Path) -> list[Path]:
    """디렉터리 있으면 그 경로 하나(통째로 삭제)."""
    return [path] if path.exists() else []


def _iter_ply(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in root.rglob("*.ply"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def targets_yolo() -> list[Path]:
    raw = YOLO / "datasets" / "raw"
    items: list[Path] = []
    items.extend(_collect_files(raw / "images", ("*.jpg", "*.jpeg", "*.png")))
    items.extend(_collect_files(raw / "labels", ("*.txt", "*.cache")))
    items.extend(_collect_tree(raw / "labels_seg"))
    items.extend(_collect_tree(YOLO / "datasets" / "splits"))
    items.extend(_collect_tree(YOLO / "datasets" / "seg"))
    items.extend(_collect_tree(YOLO / "datasets" / "custom"))
    items.extend(_collect_tree(YOLO / "runs"))
    for name in ("best.pt", "best-seg.pt"):
        pt = YOLO / "weights" / name
        if pt.is_file():
            items.append(pt)
    items.extend(_collect_files(YOLO / "datasets", ("**/*.cache",)))
    return _unique(items)


def targets_calib() -> list[Path]:
    items: list[Path] = []
    items.extend(_collect_files(VISION / "calib_data" / "handeye_tcp", ("*.png", "*.jpg", "*.jpeg", "*.json")))
    for name in ("intrinsics.json", "eye_to_hand.json"):
        path = VISION / "calib_data" / name
        if path.is_file():
            items.append(path)
    return _unique(items)


def targets_ply() -> list[Path]:
    return _iter_ply(PROJECT)


def _unique(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        key = path.resolve() if path.exists() else path
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _delete(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if path.is_dir():
        shutil.rmtree(path)


def _confirm_delete(n: int) -> bool:
    prompt = f"위 {n}개를 삭제하려면 y 입력 (그 외는 취소): "
    try:
        raw = input(prompt).strip().lower()
    except EOFError:
        print("입력이 없어 취소했습니다.")
        return False
    if raw == "y":
        return True
    print("취소했습니다.")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="교육 산출물 초기화. 목록 확인 후 y 를 입력해야 삭제."
    )
    parser.add_argument("--yolo", action="store_true", help="이미지·라벨·splits·runs·best.pt")
    parser.add_argument(
        "--calib",
        action="store_true",
        help="get_intrinsic + 손눈 샘플 + compute 결과(eye_to_hand.json)",
    )
    parser.add_argument("--ply", action="store_true", help="프로젝트 안 ply")
    parser.add_argument("--all", action="store_true", help="--yolo --calib --ply")
    args = parser.parse_args()

    yolo = args.yolo or args.all
    calib = args.calib or args.all
    ply = args.ply or args.all
    if not (yolo or calib or ply):
        parser.error("--yolo, --calib, --ply, --all 중 하나를 주세요.")

    selected: list[tuple[str, list[Path]]] = []
    if yolo:
        selected.append(("yolo", targets_yolo()))
    if calib:
        selected.append(("calib", targets_calib()))
    if ply:
        selected.append(("ply", targets_ply()))

    all_paths = _unique([p for _, group in selected for p in group])
    if not all_paths:
        print("지울 산출물이 없습니다.")
        if yolo:
            _restore_data_yaml()
            print(f"복구 {YOLO / 'data.yaml'}")
        return

    for name, group in selected:
        print(f"[{name}] {len(group)}개")
        for path in group:
            rel = path.relative_to(PROJECT) if path.is_relative_to(PROJECT) else path
            print(f"  {rel}")

    if not args.yes:
        print("삭제하지 않았습니다. 지우려면 같은 인자에 -y 를 붙이세요.")
        return

    failed = 0
    deleted = 0
    for path in all_paths:
        if not path.exists() and not path.is_symlink():
            continue
        try:
            _delete(path)
            deleted += 1
        except OSError as exc:
            print(f"실패 {path}: {exc}", file=sys.stderr)
            failed += 1
    if yolo:
        _restore_data_yaml()
        print(f"복구 {YOLO / 'data.yaml'}")
    print(f"삭제 {deleted}개" + (f"  실패 {failed}개" if failed else ""))
    if calib:
        print("다음 교육: python vision/check/get_intrinsic.py  →  python vision/handeye/capture.py  →  python vision/handeye/compute.py")


if __name__ == "__main__":
    main()
