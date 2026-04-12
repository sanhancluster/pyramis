from typing import Union
import numpy as np

from .geometry import Box, Region
from .utils.hilbert import hilbert3d, hilbert_shift_left, hilbert_add, hilbert_less_equal, HILBERT_KEY_DTYPE
from . import get_config, timer

def domain_slice(data, domain_list, bounds):
    """
    Returns a merged array of sliced portion from data based on domain_list and bounds.

    Parameters:
    data (array-like): The data to be sliced.
    domain_list (array-like): List of domain indices willing to be sliced.
    bounds (array-like): Array contains domain boundaries.
    """
    starts, ends = bounds[domain_list], bounds[domain_list+1]
    merged = np.concatenate([data[start:end] for start, end in zip(starts, ends)])
    return merged


def compute_chunk_list_from_hilbert(
        region: Union[Region, np.ndarray, list],
        hilbert_boundary,
        level_hilbert,
        boxlen: float=1.0,
        level_divide=None,
        level_subdivide: int | None=None,
        ndim: int=3,
        n_workers: int | None=None) -> np.ndarray:
    """
    Computes the list of chunk indices that intersect with the given region based on 3-dimensional Hilbert curve partitioning.

    Parameters
    ----------
    region : Region or np.ndarray
        The spatial region of interest, either as a Region instance or a (2, 3) ndarray representing a bounding box.
    hilbert_boundary : np.ndarray
        Array of Hilbert boundary keys defining the chunk partitions.
    level_hilbert : int
        The Hilbert curve level used for partitioning.
    boxlen : float
        The size of the entire box in which the Hilbert curve is defined.
    level_divide : int, optional
        The level at which to divide the Hilbert curve for chunking. If None, it is computed based on the region size.
    level_subdivide : int | None, optional
        Additional subdivision level to refine the chunking. If None, the default value from the config is used.
    n_workers : int | None, optional
        The number of workers to use for parallel computation. If None, the default value from the config is used.
    """
    timer.message("Computing chunk list from Hilbert curve...", verbose_lim=2)
    config = get_config()
    if level_subdivide is None:
        level_subdivide = int(config.get('DEFAULT_LEVEL_SUBDIVIDE', 2))
    if n_workers is None:
        n_workers = int(config.get('DEFAULT_N_WORKERS', 4))

    assert_ascending(hilbert_boundary)
    if isinstance(region, Region):
        bounding_box = region.bounding_box.box
    elif (isinstance(region, np.ndarray) or isinstance(region, list)) and np.shape(region) == (ndim, 2):
        bounding_box = np.asarray(region)
        region = Box(bounding_box)
    else:
        raise ValueError("region must be either a Region instance or a (ndim, 2) ndarray representing a bounding box.")
    
    if level_divide is None:
        minlen = np.min(bounding_box[:, 1] - bounding_box[:, 0])
        if minlen <= 0:
            return np.array([], dtype=np.int32)

        level_divide = -int(np.floor(np.log2(minlen / boxlen))) + level_subdivide
    level_divide = np.minimum(level_divide, level_hilbert)
    grid_size = boxlen * np.exp2(-level_divide)
    
    min_idx = np.floor(bounding_box[:, 0] / grid_size).astype(np.int32)
    max_idx = np.ceil(bounding_box[:, 1] / grid_size).astype(np.int32)
    timer.message(f"Using level_divide={level_divide} for chunking (grid size: {grid_size:.4f}).", verbose_lim=3)
    
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(min_idx[0], max_idx[0]),
        np.arange(min_idx[1], max_idx[1]),
        np.arange(min_idx[2], max_idx[2]),
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)

    if grid_points.shape[0] == 0:
        return np.array([], dtype=np.int32)

    if not isinstance(region, Box):
        grid_points = grid_points[region.contains((grid_points + 0.5) * grid_size, size=grid_size/2)]
    _shift = int(ndim * (level_hilbert - level_divide))
    _keys = hilbert3d(grid_points, bit_length=level_divide, n_workers=n_workers)
    hilbert_keys_min = hilbert_shift_left(_keys, _shift)
    hilbert_keys_max = hilbert_shift_left(hilbert_add(_keys, 1), _shift)
    chunk_indices_min = np.searchsorted(hilbert_boundary, hilbert_keys_min, side='right') - 1
    chunk_indices_max = np.searchsorted(hilbert_boundary, hilbert_keys_max, side='left') - 1

    if chunk_indices_max <= chunk_indices_min:
        return np.array([], dtype=np.int32)

    chunk_indices = np.unique(np.concatenate([np.arange(start, end + 1) for start, end in zip(chunk_indices_min, chunk_indices_max)]))
    timer.message(f"Found {len(chunk_indices)} chunks intersecting the region.", verbose_lim=3)
    return np.sort(chunk_indices).astype(np.int32)


def assert_ascending(arr, msg="Array is not sorted in ascending order."):
    if arr.dtype == HILBERT_KEY_DTYPE:
        ok = hilbert_less_equal(arr[:-1], arr[1:])
    else:
        ok = arr[:-1] <= arr[1:]
    if not np.all(ok):
        raise ValueError(msg)


def str_to_tuple(input_data):
    return tuple(map(int, input_data.split(',')))


def quad_to_f128(by, byteorder: str="little"):
    """
    receives byte array with format of IEEE 754 quadruple float and converts to numpy.float128 array
    because quadruple float is not supported in numpy
    source: https://stackoverflow.com/questions/52568037/reading-16-byte-fortran-floats-into-python-from-a-file
    """
    asint = []
    for raw in np.reshape(by, (-1, 16)):
        asint.append(int.from_bytes(raw, byteorder=byteorder, signed=False))
    asint = np.array(asint, dtype=object)
    sign = (np.float128(-1.0)) ** np.float128(asint >> 127)
    exponent = ((asint >> 112) & 0x7FFF) - 16383
    significand = np.float128((asint & ((1 << 112) - 1)) | (1 << 112))
    return sign * significand * 2.0 ** np.float128(exponent - 112)


def quad_to_int(by, byteorder: str = "little"):
    """
    receives byte array with format of IEEE 754 quadruple float
    and converts to python int object array
    """
    by = np.asarray(by, dtype=np.uint8).ravel()
    if by.size % 16 != 0:
        raise ValueError("Input length must be a multiple of 16 bytes")

    asint = []
    for raw in by.reshape(-1, 16):
        asint.append(int.from_bytes(raw.tobytes(), byteorder=byteorder, signed=False))

    out = []
    for bits in asint:
        sign = -1 if ((bits >> 127) & 1) else 1
        exp_bits = (bits >> 112) & 0x7FFF
        frac = bits & ((1 << 112) - 1)

        if exp_bits == 0x7FFF:
            raise ValueError("NaN or infinity cannot be converted to int")

        if exp_bits == 0:
            if frac == 0:
                out.append(0)
                continue
            mant = frac
            shift = 1 - 16383 - 112
        else:
            mant = (1 << 112) | frac
            shift = exp_bits - 16383 - 112

        val = mant << shift if shift >= 0 else mant >> (-shift)
        out.append(sign * val)

    return np.array(out, dtype=object)
