"""Start helper programs (node / Chrome / the trainer) without any window ever appearing.

On Windows a child of a console program gets its own console window unless it is started with CREATE_NO_WINDOW and
a hidden STARTUPINFO. Every place the app launches a helper uses ``hidden_kwargs()`` so none of them can open one.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Any

CREATE_NO_WINDOW = 0x08000000


def hidden_kwargs() -> dict[str, Any]:
    if sys.platform != "win32":
        return {}
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0                      # SW_HIDE
    return {"startupinfo": startup, "creationflags": CREATE_NO_WINDOW}
