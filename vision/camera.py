"""Orbbec color. 기본은 SDK + Orbbec filters 패널. V4L2는 filters=False 또는 index 지정."""

from __future__ import annotations

from pathlib import Path

import cv2

CAMERA_INDEX = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROTATE_180 = True
ORBBEC_VID = "2bc5"


def _usb_vid(sys_video: Path) -> str | None:
    device = (sys_video / "device").resolve()
    for _ in range(10):
        vid = device / "idVendor"
        if vid.is_file():
            return vid.read_text().strip().lower()
        parent = device.parent
        if parent == device:
            break
        device = parent
    return None


def find_orbbec_v4l_index() -> int | None:
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return None
    for node in sorted(root.glob("video*"), key=lambda p: p.name):
        try:
            index = int(node.name.replace("video", ""))
        except ValueError:
            continue
        if _usb_vid(node) != ORBBEC_VID:
            continue
        name = ""
        name_path = node / "name"
        if name_path.is_file():
            name = name_path.read_text().strip().lower()
        if "metadata" in name:
            continue
        return index
    return None


class OrbbecCapture:
    """detect/capture/vision_test 가 쓰는 VideoCapture 비슷한 래퍼 (Orbbec SDK)."""

    def __init__(
        self,
        width: int = FRAME_WIDTH,
        height: int = FRAME_HEIGHT,
        *,
        filters: bool = True,
        noise_filter: bool = True,
        hole_filter: bool = True,
        noise_min_diff: int | None = None,
        noise_max_size: int | None = None,
    ):
        from vision.orbbec_filters import (
            NOISE_MAX_SIZE_DEFAULT,
            NOISE_MIN_DIFF_DEFAULT,
            OrbbecFilterPanel,
        )
        from vision.rgbd import OrbbecV1

        min_diff = NOISE_MIN_DIFF_DEFAULT if noise_min_diff is None else int(noise_min_diff)
        max_size = NOISE_MAX_SIZE_DEFAULT if noise_max_size is None else int(noise_max_size)
        self._cam = OrbbecV1(
            width=width,
            height=height,
            noise_filter=noise_filter,
            noise_min_diff=min_diff,
            noise_max_size=max_size,
            hole_filter=hole_filter,
        )
        self._w = width
        self._h = height
        self._filters = None
        bgr = None
        for _ in range(30):
            bgr, _ = self._cam.grab()
            if bgr is not None:
                break
        if bgr is None:
            self._cam.close()
            raise RuntimeError(
                "Orbbec 컬러 프레임이 없습니다. Viewer를 끄고, 가능하면 USB3에 직접 연결하세요."
            )
        self._h, self._w = bgr.shape[:2]
        if filters:
            self._filters = OrbbecFilterPanel(
                self._cam,
                noise_on=noise_filter,
                min_diff=min_diff,
                max_size=max_size,
            )

    @property
    def orbbec(self):
        return self._cam

    @property
    def filter_panel(self):
        return self._filters

    def isOpened(self) -> bool:
        return getattr(self._cam, "pipe", None) is not None

    def read(self):
        if self._filters is not None:
            self._filters.sync()
        # 느린 YOLO 등으로 쌓인 프레임을 비운 뒤 최신만 쓴다.
        self._cam.flush()
        bgr, _ = self._cam.grab()
        if bgr is None:
            return False, None
        self._h, self._w = bgr.shape[:2]
        return True, bgr

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, _prop: int, _value) -> bool:
        return False

    def release(self) -> None:
        if self._filters is not None:
            self._filters.close()
            self._filters = None
        self._cam.close()


def _open_v4l2(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={index} (/dev/video{index})")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def open_camera(
    index: int | None = CAMERA_INDEX,
    *,
    filters: bool = True,
):
    """기본: Orbbec SDK + Orbbec filters 창.

    ``filters=False`` 이면 예전처럼 V4L2를 우선한다 (필터 패널 없음).
    ``index`` 를 주면 해당 V4L2만 연다.
    """
    if index is not None:
        return _open_v4l2(index)
    if filters:
        return OrbbecCapture(FRAME_WIDTH, FRAME_HEIGHT, filters=True)
    v4l = find_orbbec_v4l_index()
    if v4l is not None:
        return _open_v4l2(v4l)
    return OrbbecCapture(FRAME_WIDTH, FRAME_HEIGHT, filters=False)


def grab_bgr(cap, rotate_180: bool = ROTATE_180):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if rotate_180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def rotate_image(img, rotate_180: bool = ROTATE_180):
    if img is None or not rotate_180:
        return img
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate_rgbd(bgr, depth_mm, rotate_180: bool = ROTATE_180):
    return rotate_image(bgr, rotate_180), rotate_image(depth_mm, rotate_180)
