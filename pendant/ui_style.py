"""Pendant UI font and button defaults.

conda's Tk is built without Xft, so it never sees Noto and Korean falls back to
the bitmap font ``fixed``. The system Tk is Xft-enabled. Load that library
before tkinter, and point Tk at the matching system script directory.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

_SYS_TK = Path("/usr/lib/x86_64-linux-gnu/libtk8.6.so")
_SYS_TK_LIB = Path("/usr/share/tcltk/tk8.6")
_SYS_TCL_LIB = Path("/usr/share/tcltk/tcl8.6")

UI_FAMILY = "Noto Sans CJK KR"
MONO_FAMILY = "Noto Sans Mono CJK KR"

_enabled = False


def enable_xft_tk() -> bool:
    """Load system Xft Tk before tkinter is imported. Safe to call more than once."""
    global _enabled
    if _enabled:
        return True
    if not sys.platform.startswith("linux"):
        return False
    if not _SYS_TK.is_file() or not (_SYS_TK_LIB / "tk.tcl").is_file():
        return False
    if "tkinter" in sys.modules or "_tkinter" in sys.modules:
        return False
    os.environ["TK_LIBRARY"] = str(_SYS_TK_LIB)
    if (_SYS_TCL_LIB / "init.tcl").is_file():
        os.environ["TCL_LIBRARY"] = str(_SYS_TCL_LIB)
    ctypes.CDLL(str(_SYS_TK), mode=ctypes.RTLD_GLOBAL)
    _enabled = True
    return True


def apply_ui_theme() -> None:
    """Default CustomTkinter text to Noto Sans CJK KR and round buttons a bit more."""
    import customtkinter as ctk
    from customtkinter.windows.widgets.theme import ThemeManager

    if not ThemeManager.theme:
        ctk.set_default_color_theme("blue")
    font = ThemeManager.theme.get("CTkFont")
    if isinstance(font, dict) and "family" in font:
        font["family"] = UI_FAMILY
        font["size"] = 14
    button = ThemeManager.theme.get("CTkButton")
    if isinstance(button, dict):
        button["corner_radius"] = 10


def apply_root_fonts(root) -> None:
    """Named Tk fonts (entries, dialogs) follow the same families."""
    import tkinter as tk
    import tkinter.font as tkfont

    for name, family, size in (
        ("TkDefaultFont", UI_FAMILY, 11),
        ("TkTextFont", UI_FAMILY, 11),
        ("TkMenuFont", UI_FAMILY, 11),
        ("TkHeadingFont", UI_FAMILY, 12),
        ("TkCaptionFont", UI_FAMILY, 12),
        ("TkSmallCaptionFont", UI_FAMILY, 10),
        ("TkIconFont", UI_FAMILY, 10),
        ("TkTooltipFont", UI_FAMILY, 10),
        ("TkFixedFont", MONO_FAMILY, 11),
    ):
        try:
            tkfont.nametofont(name).configure(family=family, size=size)
        except tk.TclError:
            pass
    root.option_add("*Font", f"{{{UI_FAMILY}}} 11")


def ui_font(size: int | None = None, weight: str = "normal"):
    import customtkinter as ctk

    kwargs: dict = {"family": UI_FAMILY, "weight": weight}
    if size is not None:
        kwargs["size"] = size
    return ctk.CTkFont(**kwargs)


def mono_font(size: int = 14):
    import customtkinter as ctk

    return ctk.CTkFont(family=MONO_FAMILY, size=size)
