"""ctypes binding for the Mojo transition kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.environ.get("MOJOPYTZ_LIB") or os.path.join(
    ROOT, "dist", "libmojo-pytz.so"
)

ADDRESS = ctypes.c_void_p
I64 = ctypes.c_int64
I8 = ctypes.c_int8


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    sources = [os.path.join(ROOT, "src", "capi.mojo")]
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= max(os.path.getmtime(path) for path in sources)
    ):
        return LIB
    if os.environ.get("MOJOPYTZ_LIB"):
        if os.path.exists(LIB):
            return LIB
        raise BuildError(f"MOJOPYTZ_LIB does not exist: {LIB}")
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_LIBRARY: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _LIBRARY
    if _LIBRARY is None:
        _LIBRARY = ctypes.CDLL(build())
        _LIBRARY.mptz_resolve_utc.argtypes = [ADDRESS, I64, ADDRESS, I64, ADDRESS]
        _LIBRARY.mptz_resolve_utc.restype = None
        _LIBRARY.mptz_resolve_local.argtypes = [
            ADDRESS,
            ADDRESS,
            ADDRESS,
            I64,
            ADDRESS,
            I64,
            I8,
            I64,
            ADDRESS,
            ADDRESS,
        ]
        _LIBRARY.mptz_resolve_local.restype = None
    return _LIBRARY


def addr(
    array: np.ndarray,
    dtype: np.dtype,
    *,
    writable: bool = False,
) -> ctypes.c_void_p:
    """Return a checked address for a synchronous ctypes call."""
    expected = np.dtype(dtype)
    if not isinstance(array, np.ndarray) or array.dtype != expected:
        raise TypeError(f"expected a NumPy array with dtype {expected}")
    if not array.flags.c_contiguous or not array.flags.aligned:
        raise ValueError("FFI arrays must be C-contiguous and aligned")
    if writable and not array.flags.writeable:
        raise ValueError("FFI output arrays must be writable")
    if array.size == 0:
        raise ValueError("empty arrays have no FFI address")
    address = int(array.ctypes.data)
    if address == 0:
        raise ValueError("NumPy returned a null data pointer")
    return ctypes.c_void_p(address)
