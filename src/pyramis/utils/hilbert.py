from collections.abc import Iterable
import numpy as np
import numbers
from .. import uniform_digitize
from . import run_mp_executor

# HDF5-storable compound dtype for uint128 Hilbert keys.
# Julia: reinterpret or field access; Python: hilbert_from_compound for legacy loading.
HILBERT_KEY_DTYPE = np.dtype([('hi', np.uint64), ('lo', np.uint64)])

_TWO_64 = 1 << 64
_MASK_LO = _TWO_64 - 1


def hilbert_shift_left(arr: np.ndarray, n: int) -> np.ndarray:
    """Left shift all keys in a HILBERT_KEY_DTYPE array by n bits (scalar shift)."""
    result = np.empty(len(arr), dtype=HILBERT_KEY_DTYPE)
    if n == 0:
        result['hi'] = arr['hi']
        result['lo'] = arr['lo']
    elif n < 64:
        n64, c64 = np.uint64(n), np.uint64(64 - n)
        result['hi'] = (arr['hi'] << n64) | (arr['lo'] >> c64)
        result['lo'] = arr['lo'] << n64
    elif n < 128:
        n64 = np.uint64(n - 64)
        result['hi'] = arr['lo'] << n64
        result['lo'] = np.zeros(len(arr), dtype=np.uint64)
    else:
        result['hi'] = np.zeros(len(arr), dtype=np.uint64)
        result['lo'] = np.zeros(len(arr), dtype=np.uint64)
    return result


def hilbert_add(arr: np.ndarray, add: int) -> np.ndarray:
    """Add a specified value to each key in a HILBERT_KEY_DTYPE array."""
    result = np.empty(len(arr), dtype=HILBERT_KEY_DTYPE)
    lo_new = arr['lo'] + np.uint64(add)
    carry = (lo_new < arr['lo']).astype(np.uint64)
    result['lo'] = lo_new
    result['hi'] = arr['hi'] + carry
    return result


def hilbert_less_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a <= b for HILBERT_KEY_DTYPE arrays (lexicographic on hi, then lo)."""
    return (a['hi'] < b['hi']) | ((a['hi'] == b['hi']) & (a['lo'] <= b['lo']))


def hilbert_to_compound(ints):
    """Convert Python int iterable/array to HILBERT_KEY_DTYPE compound array."""
    scalar_input = isinstance(ints, int)
    ints = np.asarray([ints] if scalar_input else ints, dtype=object)
    result = np.empty(len(ints), dtype=HILBERT_KEY_DTYPE)
    result['hi'] = np.array([int(k) >> 64 for k in ints], dtype=np.uint64)
    result['lo'] = np.array([int(k) & _MASK_LO for k in ints], dtype=np.uint64)
    return result[0] if scalar_input else result


def hilbert_from_compound(arr: np.ndarray) -> np.ndarray:
    """Convert HILBERT_KEY_DTYPE structured array to Python int object array.
    Used only for legacy HDF5 files stored as float128.
    """
    scalar_input = arr.ndim == 0
    arr = np.atleast_1d(arr)
    hi = arr['hi'].astype(object)
    lo = arr['lo'].astype(object)
    result = hi * _TWO_64 + lo
    return result[0] if scalar_input else result


def _build_tables():
    """Build state transition tables for Hilbert computation."""
    vals = np.array([
         1, 2, 3, 2, 4, 5, 3, 5,
         0, 1, 3, 2, 7, 6, 4, 5,
         2, 6, 0, 7, 8, 8, 0, 7,
         0, 7, 1, 6, 3, 4, 2, 5,
         0, 9,10, 9, 1, 1,11,11,
         0, 3, 7, 4, 1, 2, 6, 5,
         6, 0, 6,11, 9, 0, 9, 8,
         2, 3, 1, 0, 5, 4, 6, 7,
        11,11, 0, 7, 5, 9, 0, 7,
         4, 3, 5, 2, 7, 0, 6, 1,
         4, 4, 8, 8, 0, 6,10, 6,
         6, 5, 1, 2, 7, 4, 0, 3,
         5, 7, 5, 3, 1, 1,11,11,
         4, 7, 3, 0, 5, 6, 2, 1,
         6, 1, 6,10, 9, 4, 9,10,
         6, 7, 5, 4, 1, 0, 2, 3,
        10, 3, 1, 1,10, 3, 5, 9,
         2, 5, 3, 4, 1, 6, 0, 7,
         4, 4, 8, 8, 2, 7, 2, 3,
         2, 1, 5, 6, 3, 0, 4, 7,
         7, 2,11, 2, 7, 5, 8, 5,
         4, 5, 7, 6, 3, 2, 0, 1,
        10, 3, 2, 6,10, 3, 4, 4,
         6, 1, 7, 0, 5, 2, 4, 3
    ], dtype=np.int64)

    state_diagram = vals.reshape((8, 2, 12), order='F')
    return state_diagram[:, 0, :], state_diagram[:, 1, :]


# Module-level tables (read-only, safe to share across threads)
_NSTATE_TBL, _HDIGIT_TBL = _build_tables()


def _hilbert3d_chunk(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    z_chunk: np.ndarray,
    levels_chunk: np.ndarray,
    bit_length: int,
    bl_max: int,
) -> np.ndarray:
    """Compute Hilbert keys for a chunk. Returns HILBERT_KEY_DTYPE compound array."""
    m = x_chunk.shape[0]
    # Accumulate into hi/lo uint64 pairs — no carries since each bit position is set at most once.
    order_lo = np.zeros(m, dtype=np.uint64)
    order_hi = np.zeros(m, dtype=np.uint64)
    cstate_chunk = np.zeros(m, dtype=np.int64)

    for i in range(bl_max - 1, -1, -1):
        active = levels_chunk > i
        if not np.any(active):
            continue

        b2 = ((x_chunk[active] >> i) & 1).astype(np.int64)
        b1 = ((y_chunk[active] >> i) & 1).astype(np.int64)
        b0 = ((z_chunk[active] >> i) & 1).astype(np.int64)
        sdigit = (b2 << 2) | (b1 << 1) | b0

        cs = cstate_chunk[active]
        nstate = _NSTATE_TBL[sdigit, cs]
        hdigit = _HDIGIT_TBL[sdigit, cs]

        hx = ((hdigit >> 2) & 1).astype(np.uint64)
        hy = ((hdigit >> 1) & 1).astype(np.uint64)
        hz = ((hdigit >> 0) & 1).astype(np.uint64)

        for j, bit_val in [(3*i, hz), (3*i+1, hy), (3*i+2, hx)]:
            if j < 64:
                order_lo[active] |= bit_val << np.uint64(j)
            else:
                order_hi[active] |= bit_val << np.uint64(j - 64)

        cstate_chunk[active] = nstate

    # Build compound result and apply per-element left shift (vectorized by shift range).
    result = np.empty(m, dtype=HILBERT_KEY_DTYPE)
    result['hi'] = order_hi
    result['lo'] = order_lo

    shifts = (3 * (int(bit_length) - levels_chunk)).astype(np.int64)
    hi = order_hi.copy()
    lo = order_lo.copy()

    # 0 < s < 64
    m1 = (shifts > 0) & (shifts < 64)
    if np.any(m1):
        sv = shifts[m1].astype(np.uint64)
        cv = np.uint64(64) - sv
        result['hi'][m1] = (hi[m1] << sv) | (lo[m1] >> cv)
        result['lo'][m1] = lo[m1] << sv

    # 64 <= s < 128
    m2 = (shifts >= 64) & (shifts < 128)
    if np.any(m2):
        sv = (shifts[m2] - 64).astype(np.uint64)
        result['hi'][m2] = lo[m2] << sv
        result['lo'][m2] = np.uint64(0)

    # s >= 128 (shouldn't occur with valid bit_length, but be safe)
    m3 = shifts >= 128
    if np.any(m3):
        result['hi'][m3] = np.uint64(0)
        result['lo'][m3] = np.uint64(0)

    # s == 0: already correct from the initial assignment above
    return result


def _worker_hilbert(args):
    """Top-level worker wrapper (picklable for ProcessPoolExecutor)."""
    x_chunk, y_chunk, z_chunk, levels_chunk, bit_length, bl_max = args
    return _hilbert3d_chunk(x_chunk, y_chunk, z_chunk, levels_chunk, bit_length, bl_max)


def hilbert3d(
    idx,
    bit_length: int,
    levels: int | np.ndarray | None = None,
    chunk_size: int = 1000000,
    n_workers: int = 1):
    """
    Vectorized NumPy implementation of the Fortran 'hilbert3d' subroutine.
    Returns a HILBERT_KEY_DTYPE compound array (exact, platform-independent).
    Use hilbert_shl / hilbert_add1 for arithmetic; store directly in HDF5.
    """
    idx = np.asarray(idx, dtype=np.int32)
    x = idx[:, 0]
    y = idx[:, 1]
    z = idx[:, 2]
    n = idx.shape[0]

    if levels is None:
        levels = np.full(n, bit_length, dtype=np.int32)
    elif isinstance(levels, numbers.Integral):
        levels = np.full(n, levels, dtype=np.int32)
    else:
        levels = np.asarray(levels, dtype=np.int32)
        if levels.shape[0] != n:
            raise ValueError("The first dimension of `idx` and `levels` must have the same length.")

    assert isinstance(levels, np.ndarray)
    if np.any(levels < 0):
        raise ValueError("`levels` must be non-negative.")
    if bit_length < 0:
        raise ValueError("`bit_length` must be non-negative.")

    bl_max = int(levels.max(initial=0))
    if bl_max == 0:
        return np.zeros(n, dtype=HILBERT_KEY_DTYPE)

    order = np.empty(n, dtype=HILBERT_KEY_DTYPE)

    # ----------------- single-threaded -----------------
    if n_workers in (None, 1):
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)
            order[chunk_start:chunk_end] = _hilbert3d_chunk(
                x[chunk_start:chunk_end],
                y[chunk_start:chunk_end],
                z[chunk_start:chunk_end],
                levels[chunk_start:chunk_end],
                bit_length,
                bl_max,
            )
        return order

    # ----------------- parallel (thread/process) -----------------
    tasks, slices = [], []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        slices.append((chunk_start, chunk_end))
        tasks.append(((
            x[chunk_start:chunk_end],
            y[chunk_start:chunk_end],
            z[chunk_start:chunk_end],
            levels[chunk_start:chunk_end],
            bit_length,
            bl_max,
        ),))

    results = run_mp_executor(
        _worker_hilbert,
        tasks,
        backend='thread',
        n_workers=n_workers,
        mp_method='map',
        chunksize=1
    )

    for (chunk_start, chunk_end), out_chunk in zip(slices, results):
        order[chunk_start:chunk_end] = out_chunk

    return order


def hilbert3d_map(pos: np.ndarray, bit_length: int, levels: int | np.ndarray | None = None, lims=None, check_bounds=True, kwargs_hilbert3d={}):
    """
    Position-based Hilbert curve mapping.
    Maps 3D positions to Hilbert curve indices based on the specified levels and bit length.
    """
    if lims is None:
        lims = np.array([[0, 1],] * pos.shape[-1], dtype=np.float64)

    if levels is None:
        levels = bit_length

    if isinstance(levels, Iterable):
        levels = np.asarray(levels, dtype=np.int64)
        bl_max = np.max(levels)
    elif isinstance(levels, numbers.Integral):
        bl_max = levels
        levels = np.full(pos.shape[0], levels, dtype=np.int64)
    else:
        raise ValueError("`levels` should be an integer or an array-like of integers.")

    idx = uniform_digitize(pos, lims, 2**bl_max) - 1
    if check_bounds and (np.any(idx < 0) or np.any(idx >= 2**bl_max)):
        raise ValueError("Position values out of bounds for the specified bit length.")
    if levels is not None:
        idx = idx // (2 ** (bl_max - levels))[:, np.newaxis]

    return hilbert3d(idx, bit_length, levels=levels, **kwargs_hilbert3d)
