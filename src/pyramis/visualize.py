from matplotlib.colors import AsinhNorm, LogNorm, Normalize
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from . import image, geometry
from .basic import get_position, get_unit
from .utils.arrayview import ArrayView

def plot_amr(cell: ArrayView, field=None, weight=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, coarse_bins=None, norm='auto', ax=None, **kwargs):
    cmap = seaborn.color_palette("mako", as_cmap=True) if cmap is None else cmap
    if isinstance(norm, str):
        if norm == 'log':
            norm = LogNorm(clip=True)
        elif norm == 'linear':
            norm = Normalize()
        elif norm == 'asinh':
            norm = AsinhNorm()
        elif norm == 'none':
            norm = None
        elif norm == 'auto':
            norm = 'auto'
        else:
            raise ValueError(f"Unsupported norm: {norm}")
    kw_imshow_base = dict(origin='lower', norm=norm, cmap=cmap)
    if kw_imshow is not None:
        kw_imshow_base.update(kw_imshow)
    kw_imshow = kw_imshow_base
    if isinstance(cell, ArrayView):
        region = cell.region if region is None else region
        coarse_bins = cell.info.get('coarse_bins', None) if coarse_bins is None else coarse_bins
    if region is None:
        region = geometry.Box(ndim=cell.info['ndim'])

    quantities = None if field is None else cell[field]
    weights = None if weight is None else cell[weight]
        
    im = image.amr_projection(
        get_position(cell),
        levels=cell['level'],
        quantities=quantities,
        weights=weights,
        mode=mode,
        lims=region,
        projection=projection,
        coarse_bins=coarse_bins,
        **kwargs)

    # check which normalization to use if norm is 'auto'
    if norm == 'auto':
        if np.all(im > 0):
            kw_imshow['norm'] = LogNorm(clip=True)
        else:
            kw_imshow['norm'] = Normalize()

    proj_idx = image.get_projection_index(projection)
    box = region.bounding_box
    extent = (box[proj_idx[0]][0], box[proj_idx[0]][1], box[proj_idx[1]][0], box[proj_idx[1]][1])
    if ax is None:
        ax = plt.gca()
    return ax.imshow(im, extent=extent, **kw_imshow)


def plot_part(part: ArrayView, field=None, weight=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, norm='auto', ax=None, **kwargs):
    cmap = seaborn.color_palette("mako", as_cmap=True) if cmap is None else cmap
    if isinstance(norm, str):
        if norm == 'log':
            norm = LogNorm(clip=True)
        elif norm == 'linear':
            norm = Normalize()
        elif norm == 'asinh':
            norm = AsinhNorm()
        elif norm == 'none':
            norm = None
        elif norm == 'auto':
            norm = 'auto'
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    kw_imshow_base = dict(origin='lower', norm='auto', cmap=cmap)
    if kw_imshow is not None:
        kw_imshow_base.update(kw_imshow)
    kw_imshow = kw_imshow_base
    if isinstance(part, ArrayView):
        region = part.region if region is None else region
    if region is None:
        region = geometry.Box(ndim=part.info['ndim'])
    
    quantities = None if field is None else part[field]
    weights = None if weight is None else part[weight]

    im = image.part_projection(
        get_position(part),
        quantities=quantities,
        weights=weights,
        mode=mode,
        lims=region,
        projection=projection, **kwargs)

    # check which normalization to use if norm is 'auto'
    if norm == 'auto':
        if np.all(im > 0):
            kw_imshow['norm'] = LogNorm(clip=True)
        else:
            kw_imshow['norm'] = Normalize()

    proj_idx = image.get_projection_index(projection)
    box = region.bounding_box
    extent = (box[proj_idx[0]][0], box[proj_idx[0]][1], box[proj_idx[1]][0], box[proj_idx[1]][1])
    if ax is None:
        ax = plt.gca()
    return ax.imshow(im, extent=extent, **kw_imshow)


def plot_points(part: ArrayView, projection=['x', 'y'], ax=None, **kwargs):
    proj_idx = image.get_projection_index(projection)
    pos = get_position(part)
    x = pos[proj_idx[0]]
    y = pos[proj_idx[1]]
    if ax is None:
        ax = plt.gca()
    ax.scatter(x, y, **kwargs)


def plot_circles(data, radius_name='r', projection=['x', 'y'], ax=None, color=None, **kwargs):
    proj_idx = image.get_projection_index(projection)
    pos = get_position(data)
    x = pos[proj_idx[0]]
    y = pos[proj_idx[1]]
    r = data[radius_name]
    if not isinstance(color, (list, np.ndarray)):
        color = [color] * len(x)
    if ax is None:
        ax = plt.gca()
    for xi, yi, ri, c in zip(x, y, r, color):
        circle = plt.Circle((xi, yi), ri, color=c, **kwargs)
        ax.add_patch(circle)


def add_ruler(info, region=None, ruler_size=None, projection=['x', 'y'], unit=None, fontsize=8, thickness=0.02, xy_offset=(0.075, 0.1), zorder=100, unit_options=['Gpc', 'Mpc', 'kpc', 'pc', 'au'], ax=None, color='white'):
    """
    adds horizontal ruler in current panel
    """
    if region is None:
        box = geometry.Box(ndim=info['ndim'])
    else:
        box = region.bounding_box
    proj = image.get_projection_index(projection)
    extent = box[proj[0], 1] - box[proj[0], 0]
    if unit is None:
        for u in unit_options:
            if extent / 5 >= get_unit(u, info):
                unit = u
                break
        if unit is None:
            unit = unit_options[-1]
    extent_in_unit = extent / get_unit(unit, info)
    if ruler_size is None:
        ruler_size = extent_in_unit / 5
        base = 10 ** np.floor(np.log10(ruler_size))
        ruler_size_number = ruler_size / base
        possible_numbers = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10])
        closest_number = possible_numbers[np.argmin(np.abs(possible_numbers - ruler_size_number))]
        ruler_size = closest_number * base
    bar_length = ruler_size / extent_in_unit
    rect = Rectangle(xy_offset, bar_length, thickness, transform=plt.gca().transAxes, color=color, zorder=zorder)
    if ax is None:
        ax = plt.gca()
    ax.add_patch(rect)
    ax.text(xy_offset[0] + bar_length / 2, xy_offset[1]-thickness, '%g %s' % (ruler_size, unit), ha='center',
             va='top', color=color, transform=ax.transAxes, fontsize=fontsize, zorder=zorder)