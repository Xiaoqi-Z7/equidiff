"""Expose the repository's vendored PyTorch3D rotation conversions.

The functions are copied from PyTorch3D in BIL's 3D FlowMatch dependency and
preserve EquiDiff's conversion semantics without requiring a binary PyTorch3D
wheel for the newer BIL PyTorch build.
"""

from importlib import import_module


_transforms = import_module(
    "deps.3d_flowmatch_actor.utils.pytorch3d_transforms"
)


def __getattr__(name):
    return getattr(_transforms, name)
