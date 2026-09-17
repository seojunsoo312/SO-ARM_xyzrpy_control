"""Orbbec SDK v1.10: 컬러 MJPG 640x480 + 소프트웨어 D2C 뎁스."""

from __future__ import annotations

import os
from ctypes import (
    CDLL,
    POINTER,
    RTLD_GLOBAL,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int16,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_void_p,
)
from pathlib import Path

import cv2
import numpy as np

ALIGN_D2C_SW = 2
OB_PROP_LDP_BOOL = 2  # Viewer LDP enable. 공장 기본 켜짐.
OB_PROP_DEPTH_SOFT_FILTER_BOOL = 24  # Viewer NoiseRemovalFilter
OB_PROP_DEPTH_MAX_DIFF_INT = 40  # Viewer Min Diff
OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT = 41  # Viewer Max Size
OB_PERMISSION_WRITE = 2
OB_FORMAT_MJPG = 5
OB_FORMAT_RGB = 22
OB_FORMAT_BGR = 23


def _jpeg_complete(raw: np.ndarray) -> bool:
    """SOI/EOI 없으면 USB에서 잘린 MJPEG. libjpeg 경고를 내지 않고 버린다."""
    if raw.size < 4 or int(raw[0]) != 0xFF or int(raw[1]) != 0xD8:
        return False
    tail = raw[-64:] if raw.size > 64 else raw
    return bool(np.any((tail[:-1] == 0xFF) & (tail[1:] == 0xD9)))
EP = POINTER(c_void_p)
_SDK_SO_NAMES = (
    "libOrbbecSDK.so.1.10",
    "libOrbbecSDK.so.1.10.27",
    "libOrbbecSDK.so",
)


def _sdk_so(sdk: Path) -> Path | None:
    for name in _SDK_SO_NAMES:
        path = sdk / name
        if path.is_file():
            return path
    return None


def resolve_sdk_dir() -> Path:
    env = os.environ.get("ORBBEC_SDK_DIR")
    if env:
        return Path(env).expanduser()
    downloads = Path.home() / "Downloads"
    candidates: list[Path] = []
    if downloads.is_dir():
        candidates.extend(sorted(downloads.glob("OrbbecViewer_*"), reverse=True))
    for candidate in candidates:
        if _sdk_so(candidate) is not None:
            return candidate
    return candidates[0] if candidates else downloads


def _require_sdk(sdk: Path) -> Path:
    so = _sdk_so(sdk)
    if so is None:
        raise RuntimeError(
            f"SDK 없음: {sdk}\n"
            "OrbbecViewer 폴더를 풀었으면 ORBBEC_SDK_DIR 로 그 경로를 지정하세요."
        )
    return so


def _bind(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


class OBCameraIntrinsic(Structure):
    _fields_ = [
        ("fx", c_float),
        ("fy", c_float),
        ("cx", c_float),
        ("cy", c_float),
        ("width", c_int16),
        ("height", c_int16),
    ]


class OBCameraDistortion(Structure):
    _fields_ = [
        ("k1", c_float),
        ("k2", c_float),
        ("k3", c_float),
        ("k4", c_float),
        ("k5", c_float),
        ("k6", c_float),
        ("p1", c_float),
        ("p2", c_float),
    ]


class OBD2CTransform(Structure):
    _fields_ = [
        ("rot", c_float * 9),
        ("trans", c_float * 3),
    ]


class OBCameraParam(Structure):
    _fields_ = [
        ("depthIntrinsic", OBCameraIntrinsic),
        ("rgbIntrinsic", OBCameraIntrinsic),
        ("depthDistortion", OBCameraDistortion),
        ("rgbDistortion", OBCameraDistortion),
        ("transform", OBD2CTransform),
        ("isMirrored", c_bool),
    ]


def _chk(lib, err, where: str) -> None:
    errmsg = _bind(lib, "ob_error_message", c_char_p, c_void_p)
    delerr = _bind(lib, "ob_delete_error", None, c_void_p)
    if err:
        msg = errmsg(err) or b""
        delerr(err)
        raise RuntimeError(f"{where}: {msg.decode(errors='replace')}")


def _intrinsic_dict(intr: OBCameraIntrinsic) -> dict:
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "width": int(intr.width),
        "height": int(intr.height),
    }


def _dist_dict(dist: OBCameraDistortion) -> dict:
    return {
        "k1": float(dist.k1),
        "k2": float(dist.k2),
        "k3": float(dist.k3),
        "k4": float(dist.k4),
        "k5": float(dist.k5),
        "k6": float(dist.k6),
        "p1": float(dist.p1),
        "p2": float(dist.p2),
    }


def opencv_K(intr: OBCameraIntrinsic) -> list[list[float]]:
    return [
        [float(intr.fx), 0.0, float(intr.cx)],
        [0.0, float(intr.fy), float(intr.cy)],
        [0.0, 0.0, 1.0],
    ]


def opencv_dist(dist: OBCameraDistortion) -> list[float]:
    """OpenCV Brown-Conrady: k1, k2, p1, p2, k3, k4, k5, k6."""
    return [
        float(dist.k1),
        float(dist.k2),
        float(dist.p1),
        float(dist.p2),
        float(dist.k3),
        float(dist.k4),
        float(dist.k5),
        float(dist.k6),
    ]


def _rgb_ok(param: OBCameraParam) -> bool:
    rgb = param.rgbIntrinsic
    return float(rgb.fx) >= 1.0 and int(rgb.width) > 0 and int(rgb.height) > 0


def _payload_from_param(
    param: OBCameraParam,
    *,
    method: str,
    color_w: int,
    color_h: int,
    depth_w: int,
    depth_h: int,
) -> dict:
    rgb = param.rgbIntrinsic
    K = opencv_K(rgb)
    dist = opencv_dist(param.rgbDistortion)
    rgb_info = _intrinsic_dict(rgb)
    rgb_info["distortion"] = _dist_dict(param.rgbDistortion)
    depth_info = _intrinsic_dict(param.depthIntrinsic)
    depth_info["distortion"] = _dist_dict(param.depthDistortion)
    return {
        "image_size": [int(rgb.width), int(rgb.height)],
        "camera_matrix": K,
        "dist_coeffs": dist,
        "source": "orbbec_sdk_factory",
        "sdk": "Orbbec SDK v1.10",
        "method": method,
        "requested_color": [color_w, color_h],
        "requested_depth": [depth_w, depth_h],
        "factory_camera_matrix": K,
        "factory_dist_coeffs": dist,
        "rgb": rgb_info,
        "depth": depth_info,
    }


def read_factory_intrinsics(
    color_size: tuple[int, int] = (640, 480),
    depth_size: tuple[int, int] = (640, 400),
    sdk_dir: Path | None = None,
) -> dict:
    """장치에 저장된 공장 캘리브를 SDK로 읽는다. Viewer는 끄고 실행."""
    sdk = Path(sdk_dir) if sdk_dir is not None else resolve_sdk_dir()
    so = _require_sdk(sdk)
    color_w, color_h = (int(color_size[0]), int(color_size[1]))
    depth_w, depth_h = (int(depth_size[0]), int(depth_size[1]))
    cwd = os.getcwd()
    os.chdir(sdk)
    pipe = None
    lib = None
    stop = None
    delete = None
    param_list = None
    method = ""
    param: OBCameraParam | None = None
    try:
        lib = CDLL(f"./{so.name}", mode=RTLD_GLOBAL)
        create = _bind(lib, "ob_create_pipeline", c_void_p, EP)
        get_dev = _bind(lib, "ob_pipeline_get_device", c_void_p, c_void_p, EP)
        get_list = _bind(lib, "ob_device_get_calibration_camera_param_list", c_void_p, c_void_p, EP)
        list_count = _bind(lib, "ob_camera_param_list_count", c_uint32, c_void_p, EP)
        list_get = _bind(lib, "ob_camera_param_list_get_param", OBCameraParam, c_void_p, c_uint32, EP)
        list_del = _bind(lib, "ob_delete_camera_param_list", None, c_void_p, EP)
        get_cfg = _bind(lib, "ob_pipeline_get_config", c_void_p, c_void_p, EP)
        set_align = _bind(lib, "ob_config_set_align_mode", None, c_void_p, c_int, EP)
        set_d2c = _bind(
            lib, "ob_config_set_d2c_target_resolution", None, c_void_p, c_uint32, c_uint32, EP
        )
        start = _bind(lib, "ob_pipeline_start_with_config", None, c_void_p, c_void_p, EP)
        wait = _bind(lib, "ob_pipeline_wait_for_frameset", c_void_p, c_void_p, c_uint32, EP)
        del_frame = _bind(lib, "ob_delete_frame", None, c_void_p, EP)
        get_live = _bind(lib, "ob_pipeline_get_camera_param", OBCameraParam, c_void_p, EP)
        get_profile = _bind(
            lib,
            "ob_pipeline_get_camera_param_with_profile",
            OBCameraParam,
            c_void_p,
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            EP,
        )
        stop = _bind(lib, "ob_pipeline_stop", None, c_void_p, EP)
        delete = _bind(lib, "ob_delete_pipeline", None, c_void_p, EP)

        err = c_void_p()
        pipe = create(byref(err))
        _chk(lib, err, "create_pipeline — 카메라 연결, Viewer 종료 확인")
        if not pipe:
            raise RuntimeError("pipeline 생성 실패")

        err = c_void_p()
        dev = get_dev(pipe, byref(err))
        if err:
            _chk(lib, err, "get_device")
        if dev:
            err = c_void_p()
            param_list = get_list(dev, byref(err))
            if err:
                try:
                    _chk(lib, err, "calibration_camera_param_list")
                except RuntimeError:
                    param_list = None
            if param_list:
                err = c_void_p()
                n = int(list_count(param_list, byref(err)))
                chosen = None
                for i in range(n):
                    err = c_void_p()
                    cand = list_get(param_list, i, byref(err))
                    if err:
                        continue
                    if not _rgb_ok(cand):
                        continue
                    rgb = cand.rgbIntrinsic
                    if int(rgb.width) == color_w and int(rgb.height) == color_h:
                        chosen = cand
                        break
                    if chosen is None:
                        chosen = cand
                if chosen is not None:
                    param = chosen
                    method = "ob_device_get_calibration_camera_param_list"

        if param is None or not _rgb_ok(param):
            err = c_void_p()
            cfg = get_cfg(pipe, byref(err))
            _chk(lib, err, "get_config")
            err = c_void_p()
            set_align(cfg, ALIGN_D2C_SW, byref(err))
            _chk(lib, err, "set_align")
            err = c_void_p()
            set_d2c(cfg, color_w, color_h, byref(err))
            if err:
                lib.ob_delete_error(err)
            err = c_void_p()
            start(pipe, cfg, byref(err))
            _chk(lib, err, "start — Viewer 끄기")
            for _ in range(8):
                err = c_void_p()
                fs = wait(pipe, 800, byref(err))
                if err:
                    lib.ob_delete_error(err)
                    continue
                if fs:
                    e = c_void_p()
                    del_frame(fs, byref(e))
                    break
            err = c_void_p()
            live = get_live(pipe, byref(err))
            _chk(lib, err, "get_camera_param")
            if _rgb_ok(live):
                param = live
                method = "ob_pipeline_get_camera_param"

        if param is None or not _rgb_ok(param):
            err = c_void_p()
            profile = get_profile(pipe, color_w, color_h, depth_w, depth_h, byref(err))
            _chk(lib, err, "get_camera_param_with_profile")
            param = profile
            method = "ob_pipeline_get_camera_param_with_profile"
    finally:
        if param_list and lib is not None:
            e = c_void_p()
            try:
                lib.ob_delete_camera_param_list(param_list, byref(e))
            except Exception:
                pass
        if pipe and lib is not None:
            e = c_void_p()
            try:
                if stop is not None:
                    stop(pipe, byref(e))
            except Exception:
                pass
            e = c_void_p()
            try:
                if delete is not None:
                    delete(pipe, byref(e))
            except Exception:
                pass
        os.chdir(cwd)

    if param is None or not _rgb_ok(param):
        rgb = param.rgbIntrinsic if param is not None else None
        fx = float(rgb.fx) if rgb is not None else 0.0
        wh = f"{int(rgb.width)}x{int(rgb.height)}" if rgb is not None else "0x0"
        raise RuntimeError(
            f"공장 intrinsic이 비정상입니다: fx={fx} size={wh}\n"
            "OpenNI Gemini는 스트림을 켠 뒤에만 K가 나옵니다. Viewer를 끄고 다시 실행하세요."
        )
    return _payload_from_param(
        param,
        method=method,
        color_w=color_w,
        color_h=color_h,
        depth_w=depth_w,
        depth_h=depth_h,
    )


class OrbbecV1:
    def __init__(
        self,
        width=640,
        height=480,
        sdk_dir: Path | None = None,
        *,
        noise_filter: bool | None = True,
        noise_min_diff: int | None = 51200,
        noise_max_size: int | None = 1,
        ldp: bool = False,
    ):
        sdk = Path(sdk_dir) if sdk_dir is not None else resolve_sdk_dir()
        so = _require_sdk(sdk)
        cwd = os.getcwd()
        os.chdir(sdk)
        try:
            self.lib = CDLL(f"./{so.name}", mode=RTLD_GLOBAL)
            L = self.lib
            self._create = _bind(L, "ob_create_pipeline", c_void_p, EP)
            self._get_cfg = _bind(L, "ob_pipeline_get_config", c_void_p, c_void_p, EP)
            self._set_align = _bind(L, "ob_config_set_align_mode", None, c_void_p, c_int, EP)
            self._set_d2c = _bind(
                L, "ob_config_set_d2c_target_resolution", None, c_void_p, c_uint32, c_uint32, EP
            )
            self._get_dev = _bind(L, "ob_pipeline_get_device", c_void_p, c_void_p, EP)
            self._set_bool = _bind(
                L, "ob_device_set_bool_property", None, c_void_p, c_int, c_bool, EP
            )
            self._get_bool = _bind(
                L, "ob_device_get_bool_property", c_bool, c_void_p, c_int, EP
            )
            self._set_int = _bind(
                L, "ob_device_set_int_property", None, c_void_p, c_int, c_int32, EP
            )
            self._get_int = _bind(
                L, "ob_device_get_int_property", c_int32, c_void_p, c_int, EP
            )
            self._prop_ok = _bind(
                L, "ob_device_is_property_supported", c_bool, c_void_p, c_int, c_int, EP
            )
            self._start = _bind(L, "ob_pipeline_start_with_config", None, c_void_p, c_void_p, EP)
            self._wait = _bind(
                L, "ob_pipeline_wait_for_frameset", c_void_p, c_void_p, c_uint32, EP
            )
            self._color_f = _bind(L, "ob_frameset_color_frame", c_void_p, c_void_p, EP)
            self._depth_f = _bind(L, "ob_frameset_depth_frame", c_void_p, c_void_p, EP)
            self._w = _bind(L, "ob_video_frame_width", c_uint32, c_void_p, EP)
            self._h = _bind(L, "ob_video_frame_height", c_uint32, c_void_p, EP)
            self._fmt = _bind(L, "ob_frame_format", c_int, c_void_p, EP)
            self._nbytes = _bind(L, "ob_frame_data_size", c_uint32, c_void_p, EP)
            self._data = _bind(L, "ob_frame_data", c_void_p, c_void_p, EP)
            self._scale = _bind(L, "ob_depth_frame_get_value_scale", c_float, c_void_p, EP)
            self._del_frame = _bind(L, "ob_delete_frame", None, c_void_p, EP)
            self._stop = _bind(L, "ob_pipeline_stop", None, c_void_p, EP)
            self._del_pipe = _bind(L, "ob_delete_pipeline", None, c_void_p, EP)
            self._errmsg = _bind(L, "ob_error_message", c_char_p, c_void_p)
            self._delerr = _bind(L, "ob_delete_error", None, c_void_p)

            err = c_void_p()
            self.pipe = self._create(byref(err))
            self._chk(err, "create_pipeline")
            err = c_void_p()
            self.cfg = self._get_cfg(self.pipe, byref(err))
            self._chk(err, "get_config")
            err = c_void_p()
            self._set_align(self.cfg, ALIGN_D2C_SW, byref(err))
            self._chk(err, "align")
            err = c_void_p()
            self._set_d2c(self.cfg, width, height, byref(err))
            if err:
                self._delerr(err)
            self._apply_ldp(ldp)
            self._apply_noise_filter(noise_filter, noise_min_diff, noise_max_size)
            err = c_void_p()
            self._start(self.pipe, self.cfg, byref(err))
            self._chk(err, "start — Viewer 끄기")
        finally:
            os.chdir(cwd)
        self._last_bgr = None
        self._last_depth = None

    def _chk(self, err, where):
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            raise RuntimeError(f"{where}: {msg.decode(errors='replace')}")

    def _prop_supported(self, dev, prop_id: int) -> bool:
        err = c_void_p()
        ok = bool(self._prop_ok(dev, prop_id, OB_PERMISSION_WRITE, byref(err)))
        if err:
            self._delerr(err)
            return False
        return ok

    def _set_bool_prop(self, dev, prop_id: int, value: bool, name: str) -> bool:
        if not self._prop_supported(dev, prop_id):
            print(f"{name}: 이 장치는 지원하지 않음")
            return False
        err = c_void_p()
        self._set_bool(dev, prop_id, bool(value), byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"{name}: 설정 실패 ({msg.decode(errors='replace')})")
            return False
        return True

    def _set_int_prop(self, dev, prop_id: int, value: int, name: str) -> bool:
        if not self._prop_supported(dev, prop_id):
            print(f"{name}: 이 장치는 지원하지 않음")
            return False
        err = c_void_p()
        self._set_int(dev, prop_id, int(value), byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"{name}: 설정 실패 ({msg.decode(errors='replace')})")
            return False
        return True

    def _get_bool_prop(self, dev, prop_id: int) -> bool | None:
        err = c_void_p()
        value = bool(self._get_bool(dev, prop_id, byref(err)))
        if err:
            self._delerr(err)
            return None
        return value

    def _get_int_prop(self, dev, prop_id: int) -> int | None:
        err = c_void_p()
        value = int(self._get_int(dev, prop_id, byref(err)))
        if err:
            self._delerr(err)
            return None
        return value

    def _apply_ldp(self, enable: bool) -> None:
        """Laser Distance Protection. 켜면 가까울 때 프로젝터를 끔. 기본 끔."""
        dev = self._device_ptr()
        if not dev:
            print("LDP: device 없음")
            return
        ok = self._set_bool_prop(dev, OB_PROP_LDP_BOOL, bool(enable), "LDP")
        if not ok:
            return
        cur = self._get_bool_prop(dev, OB_PROP_LDP_BOOL)
        print(f"LDP on={cur}")

    def _device_ptr(self):
        err = c_void_p()
        dev = self._get_dev(self.pipe, byref(err))
        if err:
            self._delerr(err)
            return None
        return dev

    def get_noise_filter(self) -> dict:
        dev = self._device_ptr()
        if not dev:
            return {"on": None, "min_diff": None, "max_size": None}
        return {
            "on": self._get_bool_prop(dev, OB_PROP_DEPTH_SOFT_FILTER_BOOL),
            "min_diff": self._get_int_prop(dev, OB_PROP_DEPTH_MAX_DIFF_INT),
            "max_size": self._get_int_prop(dev, OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT),
        }

    def set_noise_filter(
        self,
        enable: bool | None = True,
        min_diff: int | None = None,
        max_size: int | None = None,
        *,
        quiet: bool = False,
    ) -> dict:
        self._apply_noise_filter(enable, min_diff, max_size, quiet=quiet)
        return self.get_noise_filter()

    def _apply_noise_filter(
        self,
        enable: bool | None,
        min_diff: int | None,
        max_size: int | None,
        *,
        quiet: bool = False,
    ) -> None:
        if enable is None and min_diff is None and max_size is None:
            return
        dev = self._device_ptr()
        if not dev:
            print("NoiseRemovalFilter: device 없음")
            return
        on = False if enable is False else True
        if enable is not None or min_diff is not None or max_size is not None:
            self._set_bool_prop(dev, OB_PROP_DEPTH_SOFT_FILTER_BOOL, on, "NoiseRemovalFilter")
        if on and min_diff is not None:
            self._set_int_prop(dev, OB_PROP_DEPTH_MAX_DIFF_INT, min_diff, "Min Diff")
        if on and max_size is not None:
            self._set_int_prop(
                dev, OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, max_size, "Max Size"
            )
        if quiet:
            return
        cur = self.get_noise_filter()
        print(
            "NoiseRemovalFilter "
            f"on={cur['on']}  min_diff={cur['min_diff']}  max_size={cur['max_size']}"
        )

    def grab(self, timeout_ms=1000):
        err = c_void_p()
        fs = self._wait(self.pipe, timeout_ms, byref(err))
        if err:
            self._delerr(err)
        if not fs:
            return self._last_bgr, self._last_depth
        e = c_void_p()
        color = self._color_f(fs, byref(e))
        e = c_void_p()
        depth = self._depth_f(fs, byref(e))
        if color:
            img = self._color_to_bgr(color)
            if img is not None:
                self._last_bgr = img
            e = c_void_p()
            self._del_frame(color, byref(e))
        if depth:
            self._last_depth = self._depth_to_mm(depth)
            e = c_void_p()
            self._del_frame(depth, byref(e))
        e = c_void_p()
        self._del_frame(fs, byref(e))
        return self._last_bgr, self._last_depth

    def _color_to_bgr(self, frame):
        e = c_void_p()
        w, h = int(self._w(frame, byref(e))), int(self._h(frame, byref(e)))
        e = c_void_p()
        fmt = int(self._fmt(frame, byref(e)))
        e = c_void_p()
        n = int(self._nbytes(frame, byref(e)))
        e = c_void_p()
        ptr = self._data(frame, byref(e))
        raw = np.ctypeslib.as_array((c_uint8 * n).from_address(ptr)).copy()
        if fmt == OB_FORMAT_MJPG:
            if not _jpeg_complete(raw):
                return None
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if fmt == OB_FORMAT_RGB:
            return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == OB_FORMAT_BGR:
            return raw.reshape(h, w, 3).copy()
        return None

    def _depth_to_mm(self, frame):
        e = c_void_p()
        w, h = int(self._w(frame, byref(e))), int(self._h(frame, byref(e)))
        e = c_void_p()
        n = int(self._nbytes(frame, byref(e)))
        e = c_void_p()
        scale = float(self._scale(frame, byref(e)))
        e = c_void_p()
        ptr = self._data(frame, byref(e))
        raw = np.ctypeslib.as_array((c_uint16 * (n // 2)).from_address(ptr)).copy()
        return raw.reshape(h, w).astype(np.float32) * scale

    def close(self):
        if getattr(self, "pipe", None):
            e = c_void_p()
            self._stop(self.pipe, byref(e))
            e = c_void_p()
            self._del_pipe(self.pipe, byref(e))
            self.pipe = None
