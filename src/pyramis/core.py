from typing import Union
import numpy as np

from .geometry import Box, Region
from .utils import hilbert3d
from . import config

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


def compute_chunk_list_from_hilbert(region: Union[Region, np.ndarray, list], hilbert_boundary, level_hilbert, boxlen: float=1.0, level_divide=None, level_subdivide: int=config['DEFAULT_LEVEL_SUBDIVIDE'], ndim: int=3) -> np.ndarray:
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
    boxsize : float
        The size of the entire box in which the Hilbert curve is defined.
    level_divide : int, optional
        The level at which to divide the Hilbert curve for chunking. If None, it is computed based on the region size.
    level_subdivide : int
        Additional subdivision level to refine the chunking.
    """
    assert_ascending(hilbert_boundary)
    if isinstance(region, Region):
        bounding_box = region.bounding_box.box
    elif (isinstance(region, np.ndarray) or isinstance(region, list)) and np.shape(region) == (ndim, 2):
        bounding_box = region
        region = Box(bounding_box)
    else:
        raise ValueError("region must be either a Region instance or a (ndim, 2) ndarray representing a bounding box.")
    
    if level_divide is None:
        level_divide = -int(np.floor(np.log2(np.min(bounding_box[:, 1] - bounding_box[:, 0]) / boxlen))) + level_subdivide
    level_divide = np.minimum(level_divide, level_hilbert)
    grid_size = boxlen * np.exp2(-level_divide)
    
    min_idx = np.floor(bounding_box[:, 0] / grid_size).astype(np.int64)
    max_idx = np.ceil(bounding_box[:, 1] / grid_size).astype(np.int64)
    
    grid_x, grid_y, grid_z = np.meshgrid(
        np.arange(min_idx[0], max_idx[0]),
        np.arange(min_idx[1], max_idx[1]),
        np.arange(min_idx[2], max_idx[2]),
    )
    grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=-1)

    if not isinstance(region, Box):
        grid_points = grid_points[region.contains((grid_points + 0.5) * grid_size, size=grid_size/2)]
    hilbert_keys_min = hilbert3d(grid_points, bit_length=level_divide) * np.exp2(ndim * (level_hilbert - level_divide))
    hilbert_keys_max = (hilbert3d(grid_points, bit_length=level_divide) + 1) * np.exp2(ndim * (level_hilbert - level_divide))
    chunk_indices_min = np.searchsorted(hilbert_boundary, hilbert_keys_min, side='right') - 1
    chunk_indices_max = np.searchsorted(hilbert_boundary, hilbert_keys_max, side='left') - 1

    chunk_indices = np.unique(np.concatenate([np.arange(start, end + 1) for start, end in zip(chunk_indices_min, chunk_indices_max)]))
    return np.sort(chunk_indices)


def assert_ascending(arr, msg="Array is not sorted in ascending order."):
    if not np.all(arr[:-1] <= arr[1:]):
        raise ValueError(msg)


def str_to_tuple(input_data):
    return tuple(map(int, input_data.split(',')))


def quad_to_f16(by):
    # receives byte array with format of IEEE 754 quadruple float and converts to numpy.float128 array
    # because quadruple float is not supported in numpy
    # source: https://stackoverflow.com/questions/52568037/reading-16-byte-fortran-floats-into-python-from-a-file
    out = []
    asint = []
    for raw in np.reshape(by, (-1, 16)):
        asint.append(int.from_bytes(raw, byteorder='little'))
    asint = np.array(asint)
    sign = (np.float128(-1.0)) ** np.float128(asint >> 127)
    exponent = ((asint >> 112) & 0x7FFF) - 16383
    significand = np.float128((asint & ((1 << 112) - 1)) | (1 << 112))
    return sign * significand * 2.0 ** np.float128(exponent - 112)
