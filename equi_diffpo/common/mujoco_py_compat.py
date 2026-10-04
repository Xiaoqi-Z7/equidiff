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


def install_robosuite_egl_cleanup_shim() -> None:
    """Make robosuite's EGL context cleanup idempotent.

    Robosuite registers ``eglTerminate`` with ``atexit`` while render-context
    objects can be finalized later during interpreter shutdown.  In that
    ordering, ``EGLGLContext.free`` sees an already terminated display and
    raises ``EGL_NOT_INITIALIZED`` from ``__del__``.  The context cannot need
    further destruction once its display has terminated, so clear the stale
    handle and suppress only that cleanup-time EGL status.  Every other EGL
    error is preserved.
    """
    if os.environ.get("MUJOCO_GL", "").lower() != "egl":
        return

    try:
        from OpenGL import error as gl_error
        from robosuite.renderers.context import egl_context
    except ImportError:
        return

    context_class = egl_context.EGLGLContext
    original_free = context_class.free
    if getattr(original_free, "_bil_idempotent_cleanup", False):
        return

    def idempotent_free(self):
        try:
            original_free(self)
        except gl_error.GLError as exception:
            if getattr(exception, "err", None) != egl_context.EGL.EGL_NOT_INITIALIZED:
                raise
            self._context = None

    idempotent_free._bil_idempotent_cleanup = True
    context_class.free = idempotent_free
