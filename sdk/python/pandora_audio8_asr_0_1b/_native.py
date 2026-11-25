"""ctypes wrapper. Model work is performed by libaudio8.so, without Python tokenizers."""
import ctypes as C
import os
from importlib.resources import files
from pathlib import Path

from pandora_runtime import load_library as load_runtime_library

from ._types import ASRInferenceError

_provider_pid = None

def _check_provider():
    global _provider_pid
    if _provider_pid is not None and _provider_pid != os.getpid():
        raise RuntimeError("Native provider inherited through fork; use spawn/exec")
    _provider_pid = os.getpid()

def _bundled_library_path():
    return Path(files(__package__).joinpath("_lib", "libaudio8.so")).resolve(strict=True)


def _library_path(library=None):
    if library is None:
        library = os.environ.get("AUDIO8_LIBRARY")
        if library is None:
            return _bundled_library_path()
    library = Path(library).expanduser().resolve(strict=True)
    return library


def _open_library(library_path=None):
    _check_provider()
    load_runtime_library()
    return C.CDLL(str(_library_path(library_path)))


library = _open_library


class _Protection(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("active_buffers", C.c_uint64), ("dump_excluded_bytes", C.c_uint64),
                ("locked_bytes", C.c_uint64), ("lock_failures", C.c_uint64),
                ("last_lock_errno", C.c_int32), ("reserved", C.c_uint32)]

def protection_stats(*, library=None):
    """SDK-owned protected allocations only; no addresses or plaintext are exposed."""
    lib = _open_library(library)
    lib.a8_get_protection_stats.argtypes=[C.POINTER(_Protection)]
    lib.a8_get_protection_stats.restype=C.c_int
    s = _Protection()
    s.struct_size = C.sizeof(s)
    s.abi_version = 1
    rc = lib.a8_get_protection_stats(C.byref(s))
    if rc:
        raise ASRInferenceError("Cannot read model protection statistics", native_code=rc)
    return {name:getattr(s,name) for name,_ in s._fields_ if name not in ("struct_size","abi_version","reserved")}
