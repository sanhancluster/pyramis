
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from typing import Optional
from .basic import in_jupyter


class SharedArray(np.ndarray):
    """
    A real np.ndarray subclass that keeps a strong reference to the owning SharedView.
    This ensures the underlying SharedMemory stays alive as long as the array view exists.
    """

    def __new__(cls, input_array: np.ndarray, owner: "SharedView"):
        # Create an ndarray view of the same data, but with our subclass type
        obj = np.asarray(input_array).view(cls)
        # Attach the owner reference
        obj._owner = owner
        return obj

    def __array_finalize__(self, obj):
        """
        Called whenever a new SharedArray view is created (e.g. slicing, view casting).
        'obj' is the source object from which the new view is derived.
        """
        if obj is None:
            return
        # Propagate owner reference from the source object if present
        self._owner = getattr(obj, "_owner", None)

    @property
    def owner(self) -> "SharedView":
        """Access the SharedView that holds the SharedMemory handle."""
        return self._owner


class SharedView:
    """
    ArrayView + shared memory lifetime management.
    """
    def __init__(self, shm: SharedMemory, shape, dtype, auto_cleanup: Optional[bool] = None):
        if auto_cleanup is None:
            auto_cleanup = not in_jupyter()

        self.shm = shm
        self._auto_cleanup = auto_cleanup
        self._closed = False

        # Create ndarray backed directly by shared memory buffer
        self._arr = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=self.shm.buf)

    def _ensure_open(self):
        if self._closed:
            raise ValueError("SharedView has been closed.")

    @property
    def array(self) -> np.ndarray:
        self._ensure_open()
        return self._arr

    def __repr__(self):
        return (
            f"SharedView(shm_name={self.shm.name}, "
            f"shape={None if self._arr is None else self._arr.shape}, "
            f"dtype={None if self._arr is None else self._arr.dtype})"
        )

    def __getitem__(self, key):
        self._ensure_open()

        out = self._arr[key]

        # Return scalars as-is
        if not isinstance(out, np.ndarray):
            return out

        # If result is a view (no copy), return as SharedArray to keep shm alive
        if np.shares_memory(out, self._arr):
            # out is already a view into shm; wrap/cast it as SharedArray
            return SharedArray(out, self)

        # If it's a copy (e.g. advanced indexing), return ndarray directly
        return out
    
    def __setitem__(self, key, value):
        self._ensure_open()
        self._arr[key] = value

    def __len__(self):
        self._ensure_open()
        return len(self._arr)
    
    def __array__(self, dtype=None):
        self._ensure_open()
        if dtype is not None:
            return np.array(self._arr, dtype=dtype)
        return np.array(self._arr)
    
    def __getattr__(self, name):
        self._ensure_open()
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._arr, name)

    def _finalize(self, unlink: bool = True):
        if self._closed:
            return
        self._closed = True

        # Break reference to ndarray before closing shared memory
        self._arr = None

        try:
            self.shm.close()
        except Exception:
            pass

        if unlink:
            try:
                self.shm.unlink()
            except Exception:
                pass

    def close(self, unlink: bool = True):
        self._finalize(unlink=unlink)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._auto_cleanup:
            self._finalize(unlink=True)
