from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import seaborn

from . import image, geometry
from .basic import get_position
from .utils.arrayview import ArrayView

def plot_amr(cell: ArrayView, field=None, weight=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, coarse_bins=None, **kwargs):
    cmap = seaborn.color_palette("mako", as_cmap=True) if cmap is None else cmap
    kw_imshow_base = dict(origin='lower', norm=LogNorm(clip=True), cmap=cmap)
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
    proj_idx = image.get_projection_index(projection)
    box = region.box
    extent = (box[proj_idx[0]][0], box[proj_idx[0]][1], box[proj_idx[1]][0], box[proj_idx[1]][1])
    return plt.imshow(im, extent=extent, **kw_imshow)

def plot_part(part: ArrayView, field=None, weight=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, **kwargs):
    cmap = seaborn.color_palette("mako", as_cmap=True) if cmap is None else cmap
    kw_imshow_base = dict(origin='lower', norm=LogNorm(clip=True), cmap=cmap)
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
    proj_idx = image.get_projection_index(projection)
    box = region.box
    extent = (box[proj_idx[0]][0], box[proj_idx[0]][1], box[proj_idx[1]][0], box[proj_idx[1]][1])
    return plt.imshow(im, extent=extent, **kw_imshow)
