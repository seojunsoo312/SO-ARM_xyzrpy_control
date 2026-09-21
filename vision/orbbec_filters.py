"""Orbbec SDK filter / exposure panel (same UI as roi_cloud).

Used by vision.camera.open_camera(filters=True) and yolo/pose/roi_cloud.py.
"""

from __future__ import annotations

import cv2
import numpy as np

FILTER_WIN = "Orbbec filters"
NOISE_MIN_DIFF_DEFAULT = 10000
NOISE_MAX_SIZE_DEFAULT = 1


def _clip_int(v: int, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, v)))


class OrbbecFilterPanel:
    """RGB-D 미리보기와 분리된 필터 패널. 라벨·ON/OFF·슬라이더를 한 행에 그린다."""

    W = 720
    PAD = 14
    FOOTER_PAD = 28  # Temporal Diff 아래 여유
    HEADER_H = 44
    ROW_H = 40
    TOGGLE_W = 72
    TOGGLE_H = 28
    LABEL_W = 108
    VALUE_W = 78

    def __init__(self, cam, *, noise_on: bool, min_diff: int, max_size: int):
        self.cam = cam
        self.caps = cam.depth_ui_caps()
        self.diff_lo, self.diff_hi = self.caps["min_diff_range"]
        self.size_lo, self.size_hi = self.caps["max_size_range"]
        self.exp_lo, self.exp_hi = self.caps["exposure_range"]
        self.gain_lo, self.gain_hi = self.caps["gain_range"]
        self.cexp_lo, self.cexp_hi = self.caps["color_exposure_range"]
        self.cgain_lo, self.cgain_hi = self.caps["color_gain_range"]
        tcur = cam.get_temporal_filter()
        self.w_lo, self.w_hi = tcur["weight_range"]
        self.d_lo, self.d_hi = tcur["diffscale_range"]
        hfill = cam.get_hole_filling()

        noise = cam.get_noise_filter()
        hole = cam.get_hole_filter()
        expo = cam.get_depth_exposure()
        cexpo = cam.get_color_exposure()
        self.v = {
            "noise": bool(noise_on),
            "min_diff": _clip_int(int(noise["min_diff"] or min_diff), self.diff_lo, self.diff_hi),
            "max_size": _clip_int(int(noise["max_size"] or max_size), self.size_lo, self.size_hi),
            "hole": hole is not False,
            "holefill": True,
            "holefill_mode": _clip_int(int(hfill.get("mode") or 1), 0, 2),
            "color_ae": bool(cexpo["ae"]) if self.caps["color_ae"] else None,
            "color_exposure": _clip_int(int(cexpo["exposure"] or 0), self.cexp_lo, self.cexp_hi),
            "color_gain": _clip_int(int(cexpo["gain"] or 0), self.cgain_lo, self.cgain_hi),
            "ae": bool(expo["ae"]) if self.caps["ae"] else None,
            "exposure": _clip_int(int(expo["exposure"] or 0), self.exp_lo, self.exp_hi),
            "gain": _clip_int(int(expo["gain"] or 0), self.gain_lo, self.gain_hi),
            "temporal": False,
            "t_weight": float(tcur["weight"]),
            "t_diff": float(tcur["diffscale"]),
        }
        self._hits: list[dict] = []
        self._drag: dict | None = None
        self._dirty = True
        self._applied: dict | None = None
        self._h = self._layout_height()
        flags = cv2.WINDOW_AUTOSIZE
        if hasattr(cv2, "WINDOW_GUI_NORMAL"):
            flags |= cv2.WINDOW_GUI_NORMAL
        cv2.namedWindow(FILTER_WIN, flags)
        cv2.setMouseCallback(FILTER_WIN, self._on_mouse)
        self.sync(force=True)
        print(f"{FILTER_WIN}: 클릭 ON/OFF, 슬라이더 드래그")

    def _layout_height(self) -> int:
        n = 1 + 2  # Noise header + MinDiff/MaxSize
        if self.caps["hole"]:
            n += 1
        if self.caps.get("holefill"):
            n += 2  # HoleFilling + Mode
        n += 1  # Color header
        if self.caps["color_exposure"]:
            n += 1
        if self.caps["color_gain"]:
            n += 1
        n += 1  # DepthExp header
        if self.caps["exposure"]:
            n += 1
        if self.caps["gain"]:
            n += 1
        if self.caps["temporal"]:
            n += 3
        return self.PAD + n * self.ROW_H + self.FOOTER_PAD

    def _in_rect(self, x: int, y: int, r: tuple[int, int, int, int]) -> bool:
        x0, y0, x1, y1 = r
        return x0 <= x <= x1 and y0 <= y <= y1

    def _slider_value(self, x: int, bar: tuple[int, int, int, int], lo: float, hi: float, kind: str):
        x0, _, x1, _ = bar
        t = 0.0 if x1 <= x0 else (x - x0) / float(x1 - x0)
        t = min(1.0, max(0.0, t))
        raw = lo + t * (hi - lo)
        if kind == "int":
            return _clip_int(int(round(raw)), int(lo), int(hi))
        return float(min(hi, max(lo, raw)))

    def _on_mouse(self, event, x, y, flags, _userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for hit in self._hits:
                if not self._in_rect(x, y, hit["rect"]):
                    continue
                if hit["kind"] == "toggle":
                    key = hit["key"]
                    if self.v[key] is None:
                        return
                    self.v[key] = not bool(self.v[key])
                    self._dirty = True
                    self.sync()
                    return
                if hit["kind"] == "slider":
                    self._drag = hit
                    self._set_slider(hit, x)
                    return
        elif event == cv2.EVENT_MOUSEMOVE and self._drag is not None and (flags & cv2.EVENT_FLAG_LBUTTON):
            self._set_slider(self._drag, x)
        elif event == cv2.EVENT_LBUTTONUP:
            self._drag = None

    def _set_slider(self, hit: dict, x: int) -> None:
        key = hit["key"]
        if key in ("color_exposure", "color_gain") and self.v["color_ae"]:
            self.v["color_ae"] = False
        if key in ("exposure", "gain") and self.v["ae"]:
            self.v["ae"] = False
        self.v[key] = self._slider_value(x, hit["rect"], hit["lo"], hit["hi"], hit["num"])
        self._dirty = True
        self.sync()

    def sync(self, *, force: bool = False) -> None:
        if not force and not self._dirty:
            return
        v = self.v
        prev = self._applied
        noise = (v["noise"], v["min_diff"], v["max_size"])
        if force or prev is None or prev["noise"] != noise:
            self.cam.set_noise_filter(v["noise"], v["min_diff"], v["max_size"], quiet=True)
        if self.caps["hole"] and (
            force or prev is None or prev["hole"] != v["hole"]
        ):
            self.cam.set_hole_filter(bool(v["hole"]), quiet=True)
        holefill = (v["holefill"], v["holefill_mode"])
        if self.caps.get("holefill") and (
            force or prev is None or prev.get("holefill") != holefill
        ):
            self.cam.set_hole_filling(
                bool(v["holefill"]), int(v["holefill_mode"]), quiet=True
            )
        cexpo = (v["color_ae"], v["color_exposure"], v["color_gain"])
        if force or prev is None or prev.get("cexpo") != cexpo:
            if v["color_ae"]:
                self.cam.set_color_exposure(ae=True, quiet=True)
            else:
                self.cam.set_color_exposure(
                    ae=False if self.caps["color_ae"] else None,
                    exposure=v["color_exposure"] if self.caps["color_exposure"] else None,
                    gain=v["color_gain"] if self.caps["color_gain"] else None,
                    quiet=True,
                )
        expo = (v["ae"], v["exposure"], v["gain"])
        if force or prev is None or prev["expo"] != expo:
            if v["ae"]:
                self.cam.set_depth_exposure(ae=True, quiet=True)
            else:
                self.cam.set_depth_exposure(
                    ae=False if self.caps["ae"] else None,
                    exposure=v["exposure"] if self.caps["exposure"] else None,
                    gain=v["gain"] if self.caps["gain"] else None,
                    quiet=True,
                )
        temporal = (v["temporal"], v["t_weight"], v["t_diff"])
        if self.caps["temporal"] and (
            force or prev is None or prev["temporal"] != temporal
        ):
            self.cam.set_temporal_filter(
                bool(v["temporal"]), v["t_weight"], v["t_diff"], quiet=True
            )
        self._applied = {
            "noise": noise,
            "hole": v["hole"],
            "holefill": holefill,
            "cexpo": cexpo,
            "expo": expo,
            "temporal": temporal,
        }
        self._dirty = False
        self._draw()

    def _draw_toggle(self, img, rect, on: bool | None, enabled: bool = True) -> None:
        x0, y0, x1, y1 = rect
        if on is None:
            fill, label = (50, 50, 50), "N/A"
        elif on:
            fill, label = (46, 140, 64), "ON"
        else:
            fill, label = (50, 50, 140), "OFF"
        if not enabled:
            fill = (55, 55, 55)
        cv2.rectangle(img, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(
            img,
            label,
            ((x0 + x1 - tw) // 2, (y0 + y1 + th) // 2 - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    def _draw_slider(self, img, bar, value: float, lo: float, hi: float, enabled: bool) -> None:
        x0, y0, x1, y1 = bar
        cv2.rectangle(img, (x0, y0), (x1, y1), (58, 58, 58), -1, cv2.LINE_AA)
        span = max(1e-6, hi - lo)
        t = min(1.0, max(0.0, (float(value) - lo) / span))
        fill_x = int(round(x0 + t * (x1 - x0)))
        color = (70, 170, 90) if enabled else (80, 80, 80)
        if fill_x > x0:
            cv2.rectangle(img, (x0, y0), (fill_x, y1), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x0, y0), (x1, y1), (100, 100, 100), 1, cv2.LINE_AA)
        cx = min(x1, max(x0, fill_x))
        cv2.circle(img, (cx, (y0 + y1) // 2), 7, (230, 230, 230), -1, cv2.LINE_AA)

    def _text(self, img, text, org, color=(235, 235, 235), scale=0.52, thick=1) -> None:
        cv2.putText(
            img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA
        )

    def _draw(self) -> None:
        img = np.full((self._h, self.W, 3), 32, dtype=np.uint8)
        self._hits = []
        y = self.PAD
        right = self.W - self.PAD
        toggle_x0 = right - self.TOGGLE_W
        bar_x0 = self.PAD + self.LABEL_W + self.VALUE_W
        bar_x1 = toggle_x0 - 12

        def header(title: str, key: str | None, on: bool | None, note: str = "") -> int:
            nonlocal y
            y1 = y + self.HEADER_H
            cv2.rectangle(img, (0, y), (self.W, y1), (40, 40, 40), -1)
            self._text(img, title, (self.PAD, y + 30), (250, 250, 250), 0.62, 2)
            if note:
                self._text(img, note, (self.PAD + 210, y + 30), (160, 160, 160), 0.45)
            if key is not None:
                rect = (toggle_x0, y + 8, right, y + 8 + self.TOGGLE_H)
                self._draw_toggle(img, rect, on)
                self._hits.append({"kind": "toggle", "key": key, "rect": rect})
            y = y1
            return y

        def slider(label: str, key: str, value, lo, hi, kind: str, enabled: bool, fmt: str) -> None:
            nonlocal y
            y0 = y
            y1 = y + self.ROW_H
            mid = y + 27
            color = (220, 220, 220) if enabled else (120, 120, 120)
            self._text(img, label, (self.PAD + 10, mid), color, 0.5)
            self._text(img, fmt.format(value), (self.PAD + self.LABEL_W, mid), (80, 220, 255), 0.5)
            bar = (bar_x0, y0 + 12, bar_x1, y1 - 12)
            self._draw_slider(img, bar, float(value), float(lo), float(hi), enabled)
            if enabled:
                self._hits.append(
                    {"kind": "slider", "key": key, "rect": bar, "lo": lo, "hi": hi, "num": kind}
                )
            y = y1

        v = self.v
        header("NoiseRemoval", "noise", v["noise"])
        slider("MinDiff", "min_diff", v["min_diff"], self.diff_lo, self.diff_hi, "int", v["noise"], "{}")
        slider("MaxSize", "max_size", v["max_size"], self.size_lo, self.size_hi, "int", v["noise"], "{}")
        if self.caps["hole"]:
            header("HoleFilter", "hole", v["hole"], "ON/OFF only")
        if self.caps.get("holefill"):
            header("HoleFilling", "holefill", v["holefill"], "z=0 only, holes stay")
            names = ("Top", "Nearest", "Farest")
            mode = _clip_int(int(v["holefill_mode"]), 0, 2)
            slider(
                "Mode",
                "holefill_mode",
                mode,
                0,
                2,
                "int",
                True,
                names[mode],
            )
        header("Color", "color_ae" if self.caps["color_ae"] else None, v["color_ae"], "RGB AE")
        color_on = not bool(v["color_ae"])
        if self.caps["color_exposure"]:
            slider(
                "Exposure",
                "color_exposure",
                v["color_exposure"],
                self.cexp_lo,
                self.cexp_hi,
                "int",
                color_on,
                "{}",
            )
        if self.caps["color_gain"]:
            slider(
                "Gain",
                "color_gain",
                v["color_gain"],
                self.cgain_lo,
                self.cgain_hi,
                "int",
                color_on,
                "{}",
            )
        header("DepthExp", "ae" if self.caps["ae"] else None, v["ae"], "IR")
        exp_on = not bool(v["ae"])
        if self.caps["exposure"]:
            slider("Exposure", "exposure", v["exposure"], self.exp_lo, self.exp_hi, "int", exp_on, "{}")
        if self.caps["gain"]:
            slider("Gain", "gain", v["gain"], self.gain_lo, self.gain_hi, "int", exp_on, "{}")
        if self.caps["temporal"]:
            header("Temporal", "temporal", v["temporal"], "default OFF")
            slider(
                "Weight",
                "t_weight",
                v["t_weight"],
                self.w_lo,
                self.w_hi,
                "float",
                v["temporal"],
                "{:.3f}",
            )
            slider(
                "Diff",
                "t_diff",
                v["t_diff"],
                self.d_lo,
                self.d_hi,
                "float",
                v["temporal"],
                "{:.3f}",
            )
        cv2.imshow(FILTER_WIN, img)

    def close(self) -> None:
        try:
            cv2.destroyWindow(FILTER_WIN)
        except Exception:
            pass
