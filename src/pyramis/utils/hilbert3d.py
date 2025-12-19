from collections.abc import Iterable
import numpy as np
import numbers
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from .. import uniform_digitize

def _build_tables(bl_max: int):
    """Build state transition tables and power-of-two vector for Hilbert computation."""
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
    nstate_tbl = state_diagram[:, 0, :]
    hdigit_tbl = state_diagram[:, 1, :]

    pow2 = np.exp2(np.arange(3 * bl_max, dtype=np.int64)).astype(np.float128)
    return nstate_tbl, hdigit_tbl, pow2


def _hilbert3d_chunk(
    x_chunk: np.ndarray,
    y_chunk: np.ndarray,
    z_chunk: np.ndarray,
    levels_chunk: np.ndarray,
    bit_length: int,
    bl_max: int,
    nstate_tbl: np.ndarray,
    hdigit_tbl: np.ndarray,
    pow2: np.ndarray,
) -> np.ndarray:
    """Compute Hilbert keys for a single chunk of points."""
    m = x_chunk.shape[0]
    order_chunk = np.zeros(m, dtype=np.float128)
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
        nstate = nstate_tbl[sdigit, cs]
        hdigit = hdigit_tbl[sdigit, cs]

        hx = (hdigit >> 2) & 1
        hy = (hdigit >> 1) & 1
        hz = (hdigit >> 0) & 1

        j0 = 3 * i + 0
        j1 = 3 * i + 1
        j2 = 3 * i + 2

        order_chunk[active] += (
            hz * pow2[j0] +
            hy * pow2[j1] +
            hx * pow2[j2]
        )

        cstate_chunk[active] = nstate

    shift = 3 * (int(bit_length) - levels_chunk)
    order_chunk *= np.exp2(shift, dtype=np.float128)
    return order_chunk


def _worker_hilbert(args):
    """
    Top-level worker wrapper for process/thread pools.

    Having this at module level makes it picklable for ProcessPoolExecutor.
    """
    (x_chunk, y_chunk, z_chunk, levels_chunk,
     bit_length, bl_max, nstate_tbl, hdigit_tbl, pow2) = args

    return _hilbert3d_chunk(
        x_chunk,
        y_chunk,
        z_chunk,
        levels_chunk,
        bit_length,
        bl_max,
        nstate_tbl,
        hdigit_tbl,
        pow2,
    )


def hilbert3d(
    idx,
    bit_length: int,
    levels: int | np.ndarray | None=None,
    chunk_size: int=1000000,
    n_workers: int=1,
    backend="thread",
):
    """
    Vectorized NumPy implementation of the Fortran 'hilbert3d' subroutine.
    Supports single-thread, multithread, and multiprocess execution.
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
        return np.zeros(n, dtype=np.float128)

    # Shared lookup tables
    nstate_tbl, hdigit_tbl, pow2 = _build_tables(bl_max)

    order = np.zeros(n, dtype=np.float128)

    # ----------------- single-threaded -----------------
    if n_workers in (None, 1):
        for chunk_start in range(0, n, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n)

            x_chunk = x[chunk_start:chunk_end]
            y_chunk = y[chunk_start:chunk_end]
            z_chunk = z[chunk_start:chunk_end]
            levels_chunk = levels[chunk_start:chunk_end]

            order_chunk = _hilbert3d_chunk(
                x_chunk,
                y_chunk,
                z_chunk,
                levels_chunk,
                bit_length,
                bl_max,
                nstate_tbl,
                hdigit_tbl,
                pow2,
            )
            order[chunk_start:chunk_end] = order_chunk

        return order

    # ----------------- parallel (thread/process) -----------------
    tasks = []
    slices = []
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)
        slices.append((chunk_start, chunk_end))

        x_chunk = x[chunk_start:chunk_end]
        y_chunk = y[chunk_start:chunk_end]
        z_chunk = z[chunk_start:chunk_end]
        levels_chunk = levels[chunk_start:chunk_end]

        # Pack everything needed by the worker into a single tuple
        tasks.append((
            x_chunk,
            y_chunk,
            z_chunk,
            levels_chunk,
            bit_length,
            bl_max,
            nstate_tbl,
            hdigit_tbl,
            pow2,
        ))

    Executor = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor

    with Executor(max_workers=n_workers) as ex:
        for (chunk_start, chunk_end), out_chunk in zip(slices, ex.map(_worker_hilbert, tasks)):
            order[chunk_start:chunk_end] = out_chunk

    return order


def hilbert3d_map(pos: np.ndarray, bit_length: int, levels: int | np.ndarray | None=None, lims=None, check_bounds=True, kwargs_hilbert3d={}):
    """
    Position-based Hilbert curve mapping.
    This function maps 3D positions to Hilbert curve indices based on the specified levels and bit length.
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