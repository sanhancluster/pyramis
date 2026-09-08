import numpy as np
from .config_module import get_vname
from . import get_position_names, get_cell_size

class Region():
    def __init__(self, ndim: int, domain_lims=None):
        if domain_lims is None:
            self.domain_lims = np.array([[0, 1]] * ndim)
        else:
            self.domain_lims = np.asarray(domain_lims)

    def evaluate(self, data):
        if (isinstance(data, np.ndarray) and data.shape[-1] == 3):
            return self.contains(data)

    def contains(self, points, size=0.0):
        raise NotImplementedError()
    
    def contains_data(self, data, cell: bool=False, boxlen: float=1.0):
        raise NotImplementedError()

    @property
    def center(self):
        raise NotImplementedError()

    @property
    def bounding_box(self):
        raise NotImplementedError()

    __call__ = evaluate


class UnionRegion(Region):
    def __init__(self, regions):
        self.regions = regions

    def contains(self, points, size=0.0):
        mask = np.zeros(len(points), dtype=bool)
        for region in self.regions:
            mask |= region.contains(points, size=size)
        return mask
    
    def contains_data(self, data, cell: bool=False, boxlen: float=1.0):
        mask = np.zeros(len(data), dtype=bool)
        for region in self.regions:
            mask |= region.contains_data(data, cell=cell, boxlen=boxlen)
        return mask

    @property
    def bounding_box(self) -> "Box":
        point_min = [np.inf, np.inf, np.inf]
        point_max = [-np.inf, -np.inf, -np.inf]
        for region in self.regions:
            point_min = np.minimum(point_min, region.bounding_box.box[:, 0])
            point_max = np.maximum(point_max, region.bounding_box.box[:, 1])
        return Box(np.stack([point_min, point_max], axis=-1))


class Box(Region):
    def __init__(self, box=None, center=None, extent=None, ndim=None, domain_lims=None):
        if ndim is None:
            ndim = len(center) if center is not None else (np.array(box).shape[0] if box is not None else 3)
        if box is None:
            if center is not None and extent is not None:
                self.set_center(center, extent)
            else:
                self.box = np.asarray([[0, 1]] * ndim)
        else:
            self.box = np.asarray(box)
        self.ndim = ndim
        super().__init__(ndim=ndim, domain_lims=domain_lims)

    def set_center(self, center, extent=None):
        center = np.asarray(center)
        if extent is None:
            extent = self.extent
        extent = np.asarray(extent)
        self.box = np.stack([center - extent / 2, center + extent / 2], axis=-1)

    @property
    def extent(self) -> np.ndarray:
        return self.box[:, 1] - self.box[:, 0]

    @property
    def center(self) -> np.ndarray:
        return np.mean(self.box, axis=-1)

    @property
    def bounding_box(self) -> "Box":
        return self

    def contains(self, points, size: float | np.ndarray = 0.0):
        box = self.box

        points = np.asarray(points)
        half_size = np.asarray(size) / 2

        mask = np.all(
            (box[:, 0] <= points + half_size) &
            (points - half_size <= box[:, 1]),
            axis=-1
        )
        return mask
    
    def contains_data(self, data, cell: bool=False, boxlen: float=1.0):
        box = self.box
        if cell:
            size = get_cell_size(data, boxlen=boxlen)[..., np.newaxis]
            half_size = np.asarray(size) / 2
        else:
            half_size = np.asarray(0.0)

        mask = np.ones(len(data), dtype=bool)
        for i, key in enumerate(get_position_names()):
            if np.ndim(half_size) == 0:
                h = half_size
            elif np.ndim(half_size) == 1:
                h = half_size[i]
            else:
                h = half_size[:, i] if half_size.shape[1] > 1 else half_size[:, 0]
            mask &= (box[i, 0] <= data[key] + h) & (data[key] - h <= box[i, 1])
        return mask

    def __getitem__(self, key):
        return self.box[key]
    
    def __str__(self) -> str:
        return super().__str__() + f"(box={self.box})"


class Sphere(Region):
    def __init__(self, center, radius: float, ndim=None, domain_lims=None):
        self._center = np.asarray(center)
        self.radius = radius
        self.ndim = len(center) if ndim is None else ndim
        super().__init__(ndim=self.ndim, domain_lims=domain_lims)

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def bounding_box(self) -> "Box":
        box = Box(None, ndim=self.ndim)
        box.set_center(self.center, self.radius * 2)
        return box

    def contains(self, points, size=0.0):
        center = self.center
        radius = self.radius

        points = np.asarray(points)
        half_size = np.asarray(size) / 2

        return np.linalg.norm(points - center, axis=-1) <= radius - half_size
    
    def contains_data(self, data, cell: bool=False, boxlen: float=1.0):
        center = self.center
        radius = self.radius

        half_size = get_cell_size(data, boxlen=boxlen) / 2 if cell else 0.0

        dist2 = np.zeros(len(data), dtype=float)
        for i, key in enumerate(get_position_names()):
            # closest point from the cell to the center of the sphere
            point_cell = np.clip(center[i], data[key] - half_size, data[key] + half_size)
            dist2 += (point_cell - center[i])**2
        mask = np.sqrt(dist2) <= radius - half_size
        return mask


class Spheroid(Region):
    def __init__(self, center, radii: np.ndarray, ndim=None, domain_lims=None):
        self._center = np.asarray(center)
        self.radii = np.asarray(radii)
        self.ndim = len(center) if ndim is None else ndim
        super().__init__(ndim=self.ndim, domain_lims=domain_lims)

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def bounding_box(self) -> "Box":
        box = Box(None, ndim=self.ndim)
        box.set_center(self.center, self.radii * 2)
        return box

    def contains(self, points, size=0):
        center = self.center
        radii = self.radii

        points = np.asarray(points)
        half_size = np.asarray(size) / 2

        normed = (points - center) / (radii - half_size)
        dist = np.linalg.norm(normed, axis=-1)
        mask = dist <= 1
        return mask
    
    def contains_data(self, data, cell: bool=False, boxsize: float=1.0):
        center = self.center
        radii = self.radii
        
        half_size = get_cell_size(data, boxlen=boxsize) / 2 if cell else 0.0

        # TODO: need more accurate intersection
        dist2 = np.zeros(len(data), dtype=float)
        for i, key in enumerate(get_position_names()):
            normed = (data[key] - center[i]) / (radii[i] - half_size)
            dist2 += normed**2
        mask = dist2 <= 1
        return mask