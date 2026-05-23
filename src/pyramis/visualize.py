from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np

from . import image, geometry
from .basic import get_position
from .utils.arrayview import ArrayView

def plot_amr(cell: ArrayView, quantity_name=None, weight_name=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, **kwargs):
    kw_imshow_base = dict(origin='lower', norm=LogNorm(clip=True))
    if kw_imshow is not None:
        kw_imshow_base.update(kw_imshow)
    kw_imshow = kw_imshow_base
    if isinstance(cell, ArrayView):
        region = cell.region if region is None else region
    if region is None:
        region = geometry.Box(ndim=cell.info['ndim'])

    quantities = None if quantity_name is None else cell[quantity_name]
    weights = None if weight_name is None else cell[weight_name]
        
    im = image.amr_projection(
        get_position(cell),
        levels=cell['level'],
        quantities=quantities,
        weights=weights,
        mode=mode,
        lims=region,
        projection=projection, **kwargs)
    proj_idx = image.get_projection_index(projection)
    box = region.box
    extent = (box[proj_idx[0]][0], box[proj_idx[0]][1], box[proj_idx[1]][0], box[proj_idx[1]][1])
    return plt.imshow(im, extent=extent, cmap=cmap, **kw_imshow)

def plot_part(part: ArrayView, quantity_name=None, weight_name=None, mode='sum', projection=['x', 'y'], region=None, cmap=None, kw_imshow: dict | None=None, **kwargs):
    kw_imshow_base = dict(origin='lower', norm=LogNorm(clip=True))
    if kw_imshow is not None:
        kw_imshow_base.update(kw_imshow)
    kw_imshow = kw_imshow_base
    if isinstance(part, ArrayView):
        region = part.region if region is None else region
    if region is None:
        region = geometry.Box(ndim=part.info['ndim'])
    
    quantities = None if quantity_name is None else part[quantity_name]
    weights = None if weight_name is None else part[weight_name]

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
    return plt.imshow(im, extent=extent, cmap=cmap, **kw_imshow)
