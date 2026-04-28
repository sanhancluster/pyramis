
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from typing import Optional
from . import in_jupyter
from .. import get_vname, get_unit, get_position, get_velocity, get_cell_size, get_mass
from ..astro import get_age, get_temperature


def get_field(data: np.ndarray, info: dict, field_name: str, unit: str | None=None, aexp: float | None=None):
    """
    Get a field from the data, applying unit conversion if requested.
    """
    vname = get_vname(field_name)
    if vname in data.dtype.names:
        return data[vname]
    else:
        if field_name == 'position':
            out = get_position(data)
        elif field_name == 'velocity':
            out = get_velocity(data)
        elif field_name == 'cell_size':
            out = get_cell_size(data, boxlen=info.get('boxlen', 1.0))
        elif field_name == 'mass':
            out = get_mass(data, boxlen=info.get('boxlen', 1.0))
        elif field_name == 'age':
            out = get_age(data, info)
        elif field_name == 'temperature':
            out = get_temperature(data, info)
        else:
            raise ValueError(f"Field '{field_name}' not found in data and is not a recognized custom field.")

        if unit is not None:
            if aexp is None and get_vname('aexp') in data.dtype.names:
                aexp = data[get_vname('aexp')]
            factor = get_unit(unit, info, aexp=aexp)
            out = out / factor

        return out


class SharedArray(np.ndarray):
    """
    A real np.ndarray subclass that keeps a strong reference to the owning SharedView.
    This ensures the underlying SharedMemory stays alive as long as the array view exists.
    """

    def __new__(cls, input_array: np.ndarray, owner: "ArrayView"):
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
    def owner(self) -> "ArrayView":
        """Access the SharedView that holds the SharedMemory handle."""
        return self._owner


class ArrayView:
    """
    Unified wrapper for both private numpy arrays and shared-memory-backed arrays.

    Two construction paths:
      ArrayView(arr)                        — wraps a plain np.ndarray (no shm)
      ArrayView(shm, shape, dtype)          — wraps a SharedMemory segment
    """

    def __init__(
        self,
        arr_or_shm,
        shape=None,
        dtype=None,
        auto_cleanup: Optional[bool] = None,
        info=None,
    ):
        if auto_cleanup is None:
            auto_cleanup = not in_jupyter()

        self._auto_cleanup = auto_cleanup
        self.info = info
        self._closed = False

        if isinstance(arr_or_shm, SharedMemory):
            if shape is None or dtype is None:
                raise ValueError("shape and dtype are required when wrapping SharedMemory")
            self.shm = arr_or_shm
            self._arr = np.ndarray(tuple(shape), dtype=np.dtype(dtype), buffer=self.shm.buf)
        elif isinstance(arr_or_shm, np.ndarray):
            self.shm = None
            self._arr = arr_or_shm
        else:
            raise TypeError(
                f"Expected np.ndarray or SharedMemory, got {type(arr_or_shm).__name__}"
            )

    @classmethod
    def from_array(cls, arr: np.ndarray, auto_cleanup: Optional[bool] = None) -> "ArrayView":
        return cls(arr, auto_cleanup=auto_cleanup)

    @classmethod
    def from_shm(
        cls, shm: SharedMemory, shape, dtype, auto_cleanup: Optional[bool] = None
    ) -> "ArrayView":
        return cls(shm, shape, dtype, auto_cleanup=auto_cleanup)

    @classmethod
    def _view_of(cls, arr: np.ndarray, shm: Optional[SharedMemory], info) -> "ArrayView":
        """Create a child ArrayView that is a view into an existing array/shm. Parent owns cleanup."""
        obj = cls.__new__(cls)
        obj._auto_cleanup = False
        obj.info = info
        obj._closed = False
        obj.shm = shm
        obj._arr = arr
        return obj

    def _ensure_open(self):
        if self._closed:
            raise ValueError("ArrayView has been closed.")

    @property
    def array(self) -> np.ndarray:
        self._ensure_open()
        return self._arr

    def __repr__(self):
        loc = f"shm={self.shm.name}" if self.shm is not None else "private"
        shape = None if self._arr is None else self._arr.shape
        dtype = None if self._arr is None else self._arr.dtype
        return f"ArrayView({loc}, shape={shape}, dtype={dtype})"

    def __getitem__(self, key):
        self._ensure_open()

        if isinstance(key, tuple):
            name, unit = key
            out = get_field(self._arr, self.info, field_name=name, unit=unit)
        elif isinstance(key, str):
            name = key
            out = get_field(self._arr, self.info, field_name=name)
        else:
            out = self._arr[key]

        if not isinstance(out, np.ndarray):
            return out

        # Wrap in a new ArrayView if the result preserves the same structured dtype
        if out.dtype == self._arr.dtype:
            shm = self.shm if np.shares_memory(out, self._arr) else None
            return ArrayView._view_of(out, shm, self.info)

        # Field extraction (different dtype): keep shm alive via SharedArray for contiguous views
        if self.shm is not None and np.shares_memory(out, self._arr):
            return SharedArray(out, self)

        return out

    def __setitem__(self, key, value):
        self._ensure_open()
        if isinstance(value, ArrayView):
            value = value._arr
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

        self._arr = None

        if self.shm is not None:
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
