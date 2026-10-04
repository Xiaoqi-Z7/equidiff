"""Compatibility for robomimic's stale ``mujoco_py`` exception import.

robosuite 1.4 runs on the official ``mujoco`` binding. The pinned robomimic
wrapper only retains ``mujoco_py`` to name its rollout exception type. This
shim maps that name to ``mujoco.FatalError`` without changing the simulator.
"""

from __future__ import annotations

import os
import sys
import types


def install_mujoco_py_exception_shim() -> None:
    try:
        __import__("mujoco_py")
        return
    except ModuleNotFoundError:
        pass

    import mujoco

    module = types.ModuleType("mujoco_py")
    module.builder = types.SimpleNamespace(MujocoException=mujoco.FatalError)
    sys.modules["mujoco_py"] = module

    # The installed robomimic wrapper probes EGL through a tiny optional
    # package that is not needed by robosuite itself. Preserve its contract.
    if "egl_probe" not in sys.modules:
        egl_probe = types.ModuleType("egl_probe")

        def get_available_devices():
            return [int(os.environ.get("MUJOCO_EGL_DEVICE_ID", "0"))]

        egl_probe.get_available_devices = get_available_devices
        sys.modules["egl_probe"] = egl_probe
