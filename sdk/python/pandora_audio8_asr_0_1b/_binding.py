"""The single native provider boundary; no model algorithms live here."""
import contextlib
import ctypes as C
import os
import threading

import numpy as np

from ._native import library as load_library
from ._package import ModelPackageError, resolve_model_dir
from ._runtime import Runtime
from ._types import ASRInferenceError, ASRLoadError, ASRStateError


class NativeError(ASRInferenceError):
    def __init__(self, code, message):
        super().__init__(message)
        self.native_code = code


class ModelLoadError(ASRLoadError, OSError):
    pass


class GenerationRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("input_ids", C.POINTER(C.c_int32)), ("input_count", C.c_size_t),
                ("attention_mask", C.POINTER(C.c_int32)),
                ("input_features", C.POINTER(C.c_float)),
                ("frames", C.c_uint32), ("max_new_tokens", C.c_uint32),
                ("hotword_plan", C.c_char_p), ("reserved", C.c_uint32 * 2)]


TokenCallback = C.CFUNCTYPE(C.c_int, C.c_int32, C.c_void_p)


class StreamOptions(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("on_token", TokenCallback), ("user_data", C.c_void_p),
                ("reserved", C.c_uint32 * 2)]


class BatchRequest(C.Structure):
    _fields_ = [("struct_size", C.c_uint32), ("abi_version", C.c_uint32),
                ("rows", C.POINTER(GenerationRequest)), ("count", C.c_size_t),
                ("reserved", C.c_uint32 * 2)]


def bind_library(library):
    lib = load_library(library)
    void, size, i32, f32 = C.c_void_p, C.c_size_t, C.c_int32, C.c_float
    definitions = {
        "a8_processor_create_from_model": [void, C.POINTER(void)],
        "a8_processor_destroy_owned": [C.POINTER(void)],
        "a8_processor_prepare": [void, C.POINTER(f32), size, C.c_char_p],
        "a8_processor_get_ids": [void, C.POINTER(i32), size, C.POINTER(size)],
        "a8_processor_get_features": [void, C.POINTER(f32), size, C.POINTER(size)],
        "a8_processor_special_ids": [void, C.POINTER(i32), size, C.POINTER(size)],
        "a8_processor_encode": [void, C.c_char_p, C.POINTER(i32), size, C.POINTER(size)],
        "a8_processor_symbol": [void, i32, void, size, C.POINTER(size)],
        "a8_processor_decode": [void, C.POINTER(i32), size, C.c_uint32, void, size, C.POINTER(size)],
        "a8_processor_hotword_plan": [void, C.c_char_p, C.c_int64, C.c_double, C.c_double, void, size, C.POINTER(size)],
        "a8_create_with_runtime": [C.c_char_p, void, C.POINTER(void)],
        "a8_destroy_owned": [C.POINTER(void)],
        "a8_generate_batch": [void, C.POINTER(BatchRequest), C.POINTER(void)],
        "a8_batch_result_get": [void, size, C.POINTER(i32), size, C.POINTER(size), C.POINTER(C.c_uint32)],
        "a8_batch_result_destroy_owned": [C.POINTER(void)],
        "a8_generate_prepared": [void, C.POINTER(GenerationRequest)],
        "a8_get_generation_status": [void, C.POINTER(C.c_uint32), C.POINTER(size)],
        "a8_get_tokens": [void, C.POINTER(i32), size, C.POINTER(size)],
    }
    try:
        for name, args in definitions.items():
            fn = getattr(lib, name)
            fn.argtypes, fn.restype = args, C.c_int
        lib.a8_last_error.restype = C.c_char_p
    except AttributeError as error:
        raise RuntimeError("Native library is missing required Audio8 symbols") from error
    return lib


def text(value, name, limit):
    if not isinstance(value, str):
        raise TypeError(name + " must be a string")
    payload = value.encode("utf-8")
    if b"\0" in payload or len(payload) > limit:
        raise ValueError(name + " is NUL-containing or exceeds its UTF-8 byte limit")
    return payload


def ids(value, *, ndim):
    if not isinstance(value, np.ndarray) or value.dtype not in (np.dtype("int32"), np.dtype("int64")):
        raise TypeError("Token IDs must be a NumPy int32/int64 array")
    if value.ndim != ndim or (ndim == 2 and not 1 <= value.shape[0] <= 32):
        raise ValueError("Expected one to32 token sequences")
    if np.any(value < 0) or np.any(value >= 151936):
        raise ValueError("Token ID outside vocabulary")
    return np.ascontiguousarray(value, dtype=np.int32)


class Owner:
    @classmethod
    def _load(cls, root=None, runtime=None, library=None, *, model=None):
        obj = cls.__new__(cls)
        obj._pid, obj._lock, obj._slot = os.getpid(), threading.Lock(), C.c_void_p()
        obj._processor, obj._failed = model is not None, False
        try:
            if model is not None:
                obj._lib = model._lib
                with model._operation():
                    obj._check(obj._lib.a8_processor_create_from_model(
                        model._slot, C.byref(obj._slot)
                    ))
                return obj
            model_dir = resolve_model_dir(root)
            obj._lib = bind_library(library)
            if runtime is not None and not isinstance(runtime, Runtime):
                raise TypeError("runtime must be pandora_audio8_asr_0_1b.Runtime")
            with runtime.borrow_handle() if runtime is not None else contextlib.nullcontext() as handle:
                obj._check(obj._lib.a8_create_with_runtime(
                    os.fsencode(model_dir), handle, C.byref(obj._slot)
                ))
            return obj
        except BaseException as error:
            try:
                obj.close()
            except Exception:
                pass
            if isinstance(error, ModelPackageError):
                raise ModelLoadError(str(error), native_code=-2) from error
            raise

    def _check_process(self):
        if os.getpid() != self._pid:
            raise ASRStateError("Inherited SDK object; initialize after spawn/exec", native_code=-6)

    def _check(self, rc):
        if rc == 0:
            return
        message = self._lib.a8_last_error().decode(errors="replace")
        if rc == -1:
            raise ValueError(message)
        if rc == -2:
            raise ModelLoadError(message, native_code=rc)
        if rc == -6:
            raise ASRStateError(message, native_code=rc)
        raise NativeError(rc, message)

    @contextlib.contextmanager
    def _operation(self):
        self._check_process()
        if not self._lock.acquire(blocking=False):
            raise ASRStateError("SDK object is busy; wait for its active operation", native_code=-6)
        try:
            if not self._slot.value:
                raise ASRStateError("SDK object is closed", native_code=-6)
            if self._failed:
                raise ASRStateError("Native execution failed; close and recreate the model", native_code=-6)
            yield
        finally:
            self._lock.release()

    def _array(self, fn, dtype, *args):
        n = C.c_size_t()
        self._check(fn(self._slot, *args, None, 0, C.byref(n)))
        out = np.empty(n.value, dtype=dtype)
        ctype = C.c_int32 if np.dtype(dtype) == np.dtype("int32") else C.c_float
        self._check(fn(self._slot, *args, out.ctypes.data_as(C.POINTER(ctype)), out.size, C.byref(n)))
        return out

    def _string(self, fn, *args):
        n = C.c_size_t()
        self._check(fn(self._slot, *args, None, 0, C.byref(n)))
        out = C.create_string_buffer(n.value)
        self._check(fn(self._slot, *args, out, len(out), C.byref(n)))
        return out.raw[:n.value - 1].decode("utf-8")

    def close(self):
        self._check_process()
        if not self._lock.acquire(blocking=False):
            raise ASRStateError("Wait for the active operation before closing", native_code=-6)
        try:
            if self._slot.value:
                fn = self._lib.a8_processor_destroy_owned if self._processor else self._lib.a8_destroy_owned
                self._check(fn(C.byref(self._slot)))
        finally:
            self._lock.release()

    def __enter__(self):
        with self._operation():
            return self

    def __exit__(self, *_):
        self.close()
