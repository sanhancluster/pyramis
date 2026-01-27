import numpy as np
from skimage.transform import resize, rescale, warp
from collections.abc import Sequence, Iterable

import warnings
from . import uniform_digitize, get_dim_keys
from .geometry import Box

def get_projection_index(projection=None, ndim=3, dim_keys=None):
    """
    Get the projection index for the given projection axes.
    
    Parameters:
    -----------
    projection : list of str
        Axes to project onto. Default is ['x', 'y'].
    ndim : int, optional
        Number of dimensions. Default is 3.
    dim_keys : np.ndarray, optional
        Array of name for each axes. Default is ['x', 'y', 'z'].

    Returns:
    --------
    proj : list of int
        List of indices corresponding to the projection axes.
    """
    print(get_dim_keys())
    if projection is None:
        projection = get_dim_keys()[:2]
    if dim_keys is None:
        dim_keys = get_dim_keys()[:ndim]
    dim_index = {k: i for i, k in enumerate(dim_keys)}
    ind = [dim_index[axis] for axis in projection]
    return ind


def crop(img, range, output_shape=None, subpixel=True, **kwargs):
    """
    Crop an image to a specified range.    
    """
    range = np.array(range)
    shape = np.array(img.shape)
    idx_range = shape[:, np.newaxis] * range
    idx_range = np.clip(idx_range, 0, shape[:, np.newaxis])
    crop_shape = idx_range[:, 1] - idx_range[:, 0]

    if output_shape is None:
        output_shape = np.round(crop_shape).astype(int)
    else:
        output_shape = np.array(output_shape)

    if not subpixel:
        idx_range_int = np.array(np.round(idx_range), dtype=int)
        #idxs = np.array([np.round(shape[0] * range[0] - 0.5), np.round(shape[1] * range[1] - 0.5)], dtype=int)
        img = img[idx_range_int[0, 0]:idx_range_int[0, 1], idx_range_int[1, 0]:idx_range_int[1, 1]]
        if output_shape is not None:
            img = resize(img, output_shape, **kwargs)
    else:        
        def _inverse_map(coords):
            return (coords + 0.5) * crop_shape / output_shape - 0.5 + idx_range[:, 0]

        img = warp(img, inverse_map=_inverse_map, output_shape=output_shape, mode='edge', preserve_range=True, **kwargs)

    return img


def grid_projection(centers, levels=None, quantities=None, weights=None, shape=None, lims: Box | np.ndarray | list | None=None,
                    mode='sum', plot_method='hist', projector_kwargs={}, projection=None,
                    interp_order=0, crop_mode='subpixel', output_dtype=np.float64, padding=0,
                    type='particle', lims_domain=None):
    """
    Generate a 2D projection plot of a quantity using particle or AMR data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of the cell centers or particles.
    levels : np.ndarray, optional
        Array of shape (N,) containing the refinement levels of the cells. Required for AMR data.
    quantities : np.ndarray, optional
        Array of shape (N,) containing the quantity values to be projected. Default is None.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell or particle. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is None.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.
    crop_mode : str, optional
        Mode for cropping the image to fit the output to be within desired limits. Options are 'grid', 'pixel', 'subpixel'. Default is 'subpixel'.
        'grid': crop the image based on the current minimum resolution of the grid.
        'pixel': crop the image based on the pixel size of the drawing image.
        'subpixel': crop the image with allowing subpixel cropping.
    output_dtype : dtype, optional
        Data type of the output grid. Default is np.float64.
    padding : int, optional
        Number of padding pixels in the boundary to draw the data. Default is 0.
    type : str, optional
        Type of data. Options are 'particle' or 'amr'. Default is 'particle'.
    lims_domain : list of list of float, optional
        Domain limits for the data in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]] to used for calculating cell size based on levels. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    def apply_projection(grid, grid_weight, x, y, quantity, weights, lims_2d, projector, mode='sum'):
        shape = grid.shape
        if mode in ['sum', 'mean']:
            # do a weighted sum over projection
            grid += projector(x, y, quantity*weights, lims_2d, shape)
            if mode in ['mean']:
                grid_weight += projector(x, y, weights, lims_2d, shape)
        elif mode in ['min', 'max']:
            xi = uniform_digitize(x, lims_2d[0], shape[0])
            yi = uniform_digitize(y, lims_2d[1], shape[1])

            # add 1 pixel padding to bin out of range values
            grid_padding = np.full(np.asarray(shape)+2, fill_value=np.inf if mode == 'min' else -np.inf, dtype=output_dtype)
            if mode in ['min']:
                # do a minimum over projection
                np.minimum.at(grid_padding, (xi, yi), quantity)
                np.minimum(grid, grid_padding[1:-1, 1:-1], out=grid)
            elif mode in ['max']:
                # do a maximum over projection
                np.maximum.at(grid_padding, (xi, yi), quantity)
                np.maximum(grid, grid_padding[1:-1, 1:-1], out=grid)

    ndim_data = 3
    ndim_proj = 2
    levelmin, levelmax = None, None

    if projection is None:
        projection = get_dim_keys()[:ndim_proj]

    # if lims is None, set all limits to [0, 1]
    if not isinstance(lims, Box):
        region = Box(lims)
    else:
        region = lims
    lims = region.box
    if lims is None:
        if lims_domain is None:
            lims = [[0, 1],] * ndim_data
        else:
            lims = lims_domain
    lims = np.asarray(lims)

    if lims_domain is None:
        lims_domain = np.array([[0, 1],] * ndim_data)
        lims_domain = np.stack([np.minimum(lims[:, 0], lims_domain[:, 0]), np.maximum(lims[:, 1], lims_domain[:, 1])], axis=-1)
    lims_domain = np.asarray(lims_domain)
    # if quantities is None, set all quantities to 1
    if quantities is None:
        quantities = np.ones(centers.shape[0])
    # if weights is None, set all weights to 1
    if weights is None:
        weights = np.ones_like(quantities)
    # if shape is scalar, make it a tuple
    if np.isscalar(shape):
        shape = tuple(np.repeat(shape, 2))
    # if number of arrays does not match the number of quantities, raise an error
    if centers.shape[0] != len(quantities):
        raise ValueError("The number of centers and quantities do not match.")
    # if number of arrays does not match the number of weights, raise an error
    if weights is not None and centers.shape[0] != len(weights):
        raise ValueError("The number of centers and weights do not match.")
    if plot_method != 'hist' and mode in ['min', 'max']:
        warnings.warn("plot_method is disabled when mode is min, max")

    if type in ['particle', 'part']:
        type = 'part'
    elif type in ['amr', 'cell', 'grid']:
        type = 'amr'

    proj = get_projection_index(projection, ndim=ndim_data)
    
    # get the z-axis that is not in the projection
    proj_z = np.setdiff1d(np.arange(ndim_data), proj)[0]
    lims_2d = lims[proj]

    # size of the whole domain
    domain_shape = np.array(lims_domain)[:, 1] - np.array(lims_domain)[:, 0]
    domain_length_max = np.max(domain_shape)
    lims_domain_2d = lims_domain[proj]

    # region of interest in the domain coordinates
    scope = (lims - lims_domain[:, 0, np.newaxis]) / domain_length_max
    scope_2d = scope[proj]

    if mode in ['sum', 'mean']:
        fill_value = 0
    elif mode in ['min']:
        fill_value = np.inf
    elif mode in ['max']:
        fill_value = -np.inf
    
    if type == 'part':
        if shape is None:
            shape = (100, 100)
    elif type == 'amr':
        if levels is None:
            raise ValueError("Levels must be provided for AMR data.")
        if centers.shape[0] != len(levels):
            raise ValueError("The number of centers and levels do not match.")
        levelmin, levelmax = np.min(levels), np.max(levels)
        levelmin_draw = levelmin

        if shape is None:
            # if shape is not specified, draw with the full resolution
            levelmax_draw = levelmax
            dx_min = 2. ** -levelmax_draw
            shape = (scope_2d[0, 1] - scope_2d[0, 0]) // dx_min, (scope_2d[1, 1] - scope_2d[1, 0]) // dx_min
            if np.prod(shape) >= 1E8:
                warnings.warn(f"The shape of the grid is too large: {shape}, it may cause memory issues.")
        else:
            # get the levels of the grid to draw the desired resolution
            dx_min = np.minimum((scope_2d[0, 1] - scope_2d[0, 0]) / shape[0], (scope_2d[1, 1] - scope_2d[1, 0]) / shape[1])
            levelmax_draw = np.minimum(np.ceil(-np.log2(dx_min)).astype(int), levelmax)
    else:
        raise ValueError("Unknown type: %s. Supported types are 'part' and 'amr'." % type)

    pixel_size = (lims_2d[:, 1] - lims_2d[:, 0]) / np.array(shape)
    if padding != 0:
        padding_size = np.asarray([0.]*ndim_data)
        padding_size[proj] = pixel_size * padding
    else:
        padding_size = 0.

    if type == 'part':
        # get mask that indicates particles to draw
        mask_draw = region.contains(centers, size=padding_size * 2)
        shape_grid = shape

    elif type == 'amr':
        # get the smallest levelmin grid space that covers the whole region
        i_lims_levelmin = scope_2d * 2**levelmin_draw
        i_lims_levelmin[:, 0] = np.floor(i_lims_levelmin[:, 0])
        i_lims_levelmin[:, 1] = np.ceil(i_lims_levelmin[:, 1])
        i_lims_levelmin = i_lims_levelmin.astype(int)

        shape_grid = tuple(i_lims_levelmin[:, 1] - i_lims_levelmin[:, 0])
        lims_2d_draw = (i_lims_levelmin / 2**levelmin_draw) * domain_length_max + lims_domain_2d[:, 0, np.newaxis]

        assert levels is not None
        # get mask to draw the particles that are within the limits of the current drawing scope
        # apply padding to the limits
        mask_draw = region.contains(centers, size=(0.5**levels * domain_length_max)[..., np.newaxis] + padding_size * 2)
        ll = levels[mask_draw]
    
    else:
        raise ValueError("Unknown type: %s. Supported types are 'part' and 'amr'." % type)

    # initialize grid and grid_weight
    grid = np.full(shape_grid, fill_value, dtype=output_dtype)
    if mode in ['mean']:
        grid_weight = np.zeros(shape_grid, dtype=output_dtype)
    else:
        grid_weight = None

    # apply mask based on the current drawing scope
    cc = centers[mask_draw]
    qq = quantities[mask_draw]
    ww = weights[mask_draw]

    # get the projected coordinates
    xx, yy = cc[:, proj[0]], cc[:, proj[1]]
    zz = cc[:, proj_z]

    # set the projector lambda based on the plot method
    projector = lambda x, y, w, lims, shape: density_2d(x, y, lims=lims, weights=w, shape=shape, density=False, method=plot_method, **projector_kwargs)

    # do projection
    if type == 'part':
        pixel_area = np.prod((lims_2d[:, 1] - lims_2d[:, 0]) / np.array(shape))

        # do projection onto current grid
        apply_projection(grid=grid, grid_weight=grid_weight, x=xx, y=yy, quantity=qq, weights=ww, lims_2d=lims_2d, projector=projector, mode=mode)
        
        grid /= pixel_area

        if mode == 'mean' and grid_weight is not None:
            grid /= grid_weight

    elif type == 'amr' and levelmin is not None and levelmax is not None:
        for grid_level in range(levelmin, levelmax+1):
            mask_level = ll == grid_level
            shape_now = grid.shape
            x, y, z, q = xx[mask_level], yy[mask_level], zz[mask_level], qq[mask_level]

            volume_weight = 1
            # get weight for the current level with depending on the line-of-sight depth within the limit.
            # e.g., the weight is 0.5 if the z-coordinate is in the middle of any z-limits 
            volume_weight += np.clip((z - lims[proj_z][0]) * 2**grid_level - 0.5, -1, 0) + np.clip((lims[proj_z][1] - z) * 2**grid_level - 0.5, -1, 0)
            # multiply the weight by the depth of the projected grid. The weight is doubled per each decreasing level except for the levels larger than the grid resolution.
            volume_weight *= 0.5**grid_level
            # give additional weight to the projection if cell is smaller than the grid resolution
            volume_weight *= 0.25**np.maximum(0, grid_level - levelmax_draw)
            w = ww[mask_level] * volume_weight

            # do projection onto current level grid
            apply_projection(grid=grid, grid_weight=grid_weight, x=x, y=y, quantity=q, weights=w, lims_2d=lims_2d_draw, projector=projector, mode=mode)

            # increase grid size if necessary
            if grid_level >= levelmin_draw and grid_level < levelmax_draw:
                grid = rescale(grid, 2, order=interp_order)
                if mode in ['mean']:
                    grid_weight = rescale(grid_weight, 2, order=interp_order)

        if mode == 'mean' and grid_weight is not None:
            grid /= grid_weight
        # resize and crop image to the desired shape
        if shape is not None:
            if crop_mode == 'grid':
                grid = resize(grid, output_shape=shape, order=interp_order)
            elif crop_mode in ['pixel', 'subpixel']:
                subpixel = crop_mode == 'subpixel'
                lims_crop = (lims_2d - lims_2d_draw[:, 0, np.newaxis]) / (lims_2d_draw[:, 1] - lims_2d_draw[:, 0])[:, np.newaxis]
                grid = crop(grid, range=lims_crop, output_shape=shape, subpixel=subpixel, order=interp_order)

    return grid.T


def part_projection(centers, quantities=None, weights=None, shape=100, lims=None,
                    mode='sum', plot_method='hist', projection=None, output_dtype=np.float64, padding=0, lims_domain=None):
    """
    Generate a 2D projection plot of a quantity using particle data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of particles.
    quantities : np.ndarray
        Array of shape (N,) containing the quantity values to be projected.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is 100.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    return grid_projection(centers=centers, levels=None, quantities=quantities, weights=weights, shape=shape, lims=lims, mode=mode, plot_method=plot_method, projection=projection, output_dtype=output_dtype, padding=padding, type='particle', lims_domain=lims_domain)


def amr_projection(centers, levels, quantities=None, weights=None, shape=None, lims=None,
                   mode='sum', plot_method='hist', projection=None, interp_order=0, output_dtype=np.float64, padding=0, crop_mode='subpixel', lims_domain=None):
    """
    Generate a 2D projection plot of a quantity using Adaptive Mesh Refinement (AMR) data.

    Parameters:
    -----------
    centers : np.ndarray
        Array of shape (N, 3) containing the coordinates of the cell centers. Should be within the range of (0, 1).
    levels : np.ndarray
        Array of shape (N,) containing the refinement levels of the cells.
    quantities : np.ndarray
        Array of shape (N,) containing the quantity values to be projected.
    weights : np.ndarray, optional
        Array of shape (N,) containing the weights for each cell. If None, all weights are set to 1. Default is None.
    shape : int or tuple of int, optional
        Shape of the output grid. If an integer is provided, it is used for both dimensions. Default is None.
    lims : list of list of float, optional
        Limits for the projection in the form [[xmin, xmax], [ymin, ymax], [zmin, zmax]]. If None, defaults to [[0, 1], [0, 1], [0, 1]]. Default is None.
    mode : str, optional
        Mode of projection. Options are 'sum', 'mean', 'min', 'max'. Default is 'sum'.
    plot_method : str, optional
        Method for plotting. Options are 'hist' for histogram and 'cic' for Cloud-In-Cell. Default is 'hist'.
    projection : list of str, optional
        Axes to project onto. Default is ['x', 'y'].
    interp_order : int, optional
        Order of interpolation for rescaling. Default is 0.

    Returns:
    --------
    grid : np.ndarray
        2D array representing the projected quantity.
    """
    return grid_projection(centers=centers, levels=levels, quantities=quantities, weights=weights, shape=shape, lims=lims, mode=mode, plot_method=plot_method, projection=projection, interp_order=interp_order, crop_mode=crop_mode, output_dtype=output_dtype, padding=padding, type='amr', lims_domain=lims_domain)


def density_2d(x, y, lims, shape=100, weights=None, density=False, method='hist', **kwargs):
    """
    Estimate the density of points in 2D space and return a 2D image.
    """
    # apply kde-like image convolution using gaussian filter
    x, y = np.array(x), np.array(y)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # set up shapes and ranges
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    shape_array = np.asarray(shape)
    range_array = np.asarray(lims)

    if(weights is not None):
        weights = weights[mask]
    else:
        weights = np.ones_like(x)

    if method == 'hist':
        return hist_2d(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'cic':
        return cic_2d(x, y, range_array, shape_array, weights=weights, density=density, **kwargs)
    elif method == 'hist_numpy':
        im = np.histogram2d(x, y, range=range_array, bins=shape_array, weights=weights, **kwargs)[0]
        if density:
            area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / im.size
            im /= area_per_px
        return im
    else:
        raise ValueError("Unknown mode: %s. Use 'hist', 'kde', 'cic', 'gaussian', or 'dtfe'." % method)


def hist_2d(x, y, lims, shape:int | np.ndarray | Sequence[int]=100, weights=None, density=False):
    """
    Create a 2D histogram image from x and y coordinates.
    Only works for the uniform grid and faster than np.histogram2d, but the result may slightly vary near the bin edges.
    Parameters:
    - x, y: 1D arrays of coordinates
    - lims: 2D array defining the limits of the image, shape (2, 2)
    - reso: resolution of the image, can be a scalar or a tuple/list of two values
    - weights: optional 1D array of weights for each point
    - density: if True, normalize the histogram to represent a probability density function
    - filter_sigma: if provided, apply Gaussian smoothing with this sigma value
    Returns:
    - pool: 2D numpy array representing the histogram image, shape (reso[0], reso[1])
    """
    
    # set up shapes and ranges
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(shape)
    shape_pad = shape_array + 2

    # mask calculation
    x = np.asarray(x)
    y = np.asarray(y)

    if weights is None:
        weights = np.ones_like(x)
    else:
        weights = np.asarray(weights)

    xi = uniform_digitize(x, lims[0], shape_array[0])
    yi = uniform_digitize(y, lims[1], shape_array[1])

    flat_indices = xi * shape_pad[1] + yi
    accum = np.bincount(flat_indices, weights=weights, minlength=shape_pad[0] * shape_pad[1])
    pool = accum.reshape(shape_pad)[1:-1, 1:-1]
    
    if density:
        area_per_px = (lims[0][1] - lims[0][0]) * (lims[1][1] - lims[1][0]) / pool.size
        pool /= area_per_px

    return pool


def cic_2d(x, y, lims, shape:int | np.ndarray | Sequence[int]=100, weights=None, density=False, full_vectorize=False):
    """
    Create a 2D image using Cloud-in-Cell (CIC) method.
    This method is useful for creating density maps from point data.
    Parameters:
    - x, y: 1D arrays of coordinates
    - lims: 2D array defining the limits of the image, shape (2, 2)
    - reso: resolution of the image, can be a scalar or a tuple/list of two values
    - weights: optional 1D array of weights for each point
    - full_vectorize: if True, uses a fully vectorized approach (memory-intensive)
    Returns:
    - pool: 2D numpy array representing the image, shape (reso[0], reso[1])
    Note: The function assumes that the input coordinates are finite and within the specified limits.
    If the coordinates are outside the limits, they will be ignored.
    """

    # set up shapes and ranges
    if np.isscalar(shape):
        shape = np.repeat(shape, 2)
    lims = np.asarray(lims)
    shape_array = np.asarray(shape)
    shape_pad = shape_array + 2
    range_array = np.asarray(lims)

    # mask calculation
    x = np.asarray(x)
    y = np.asarray(y)
    mask = np.isfinite(x) & np.isfinite(y)

    range_size = range_array[:, 1] - range_array[:, 0]
    range_pad = range_array + np.asarray([-0.5, 0.5]) * range_size / shape_array
    mask &= (x >= range_pad[0, 0]) & (x < range_pad[0, 1])
    mask &= (y >= range_pad[1, 0]) & (y < range_pad[1, 1])

    x = x[mask]
    y = y[mask]

    points = np.stack([x, y], axis=-1)

    if weights is None:
        weights = np.ones_like(x)
    elif isinstance(weights, Iterable):
        weights = np.asarray(weights)[mask]

    # Normalize coordinates to [0, 1] range
    indices_float = (points - range_array[:, 0]) / range_size * shape_array + 0.5
    indices_float = indices_float.reshape(-1, 2)

    # Create zero array for accumulation
    dxs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    if not full_vectorize:
        pool = np.zeros(shape_array, dtype='f8')
        for dx in dxs:
            indices_int = np.floor(indices_float - dx).astype(np.int32)
            offsets = indices_float - indices_int
            areas = (1 - np.abs(offsets[:, 0] - 1)) * (1 - np.abs(offsets[:, 1] - 1))
            values = areas * weights

            indices_int += 1  # padding offset
            flat_indices = indices_int[:, 0] * shape_pad[1] + indices_int[:, 1]
            accum = np.bincount(flat_indices, weights=values, minlength=shape_pad[0] * shape_pad[1])
            pool += accum.reshape(shape_pad)[1:-1, 1:-1]
    else:
        # full vectorization: memory-intensive
        indices_int = np.floor(indices_float[None, :, :] - dxs[:, None, :]).astype(np.int32)
        offsets = indices_float[None, :, :] - indices_int
        areas = (1 - np.abs(offsets[..., 0] - 1)) * (1 - np.abs(offsets[..., 1] - 1))
        values = (areas * weights[None, :]).reshape(-1)

        indices_int += 1
        flat_indices = (indices_int[..., 0] * shape_pad[1] + indices_int[..., 1]).reshape(-1)
        accum = np.bincount(flat_indices, weights=values, minlength=shape_pad[0] * shape_pad[1])
        pool = accum.reshape(shape_pad)[1:-1, 1:-1]
    
    if density:
        area_per_px = (range_array[0, 1] - range_array[0, 0]) * (range_array[1, 1] - range_array[1, 0]) / pool.size
        pool /= area_per_px

    return pool
