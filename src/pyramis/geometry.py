import numpy as np
from . import get_dim_keys, get_cell_size

class Region():
    def evaluate(self, data):
        if (isinstance(data, np.ndarray) and data.shape[-1] == 3):
            return self.contains(data)

    def contains(self, points, size=0.0):
        raise NotImplementedError()
    
    def contains_data(self, data, cell: bool=False, boxsize: float=1.0):
        raise NotImplementedError()

    @property
    def center(self):
        raise NotImplementedError()

    @property
    def bounding_box(self):
        raise NotImplementedError()

    __call__ = evaluate


class Box(Region):
    def __init__(self, box):
        self.box = np.asarray(box)

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

    def contains(self, points, size: float | np.ndarray=0.0):
        box = self.box
        half_size = np.asarray(size)[..., np.newaxis] / 2

        mask = np.all(
            (box[:, 0] <= points + half_size) &
            (points - half_size <= box[:, 1]),
            axis=-1
        )
        return mask
    
    def contains_data(self, data, cell: bool=False, boxsize: float=1.0):
        box = self.box
        if cell:
            size = get_cell_size(data, boxsize=boxsize)
            half_size = np.asarray(size) / 2
        else:
            half_size = 0.0

        mask = np.ones(len(data), dtype=bool)
        for i, key in enumerate(get_dim_keys()):
            mask &= (box[i, 0] <= data[key] + half_size) & (data[key] - half_size <= box[i, 1])
        return mask


    def __getitem__(self, key):
        return self.box[key]


class Sphere(Region):
    def __init__(self, center, radius: float):
        self._center = np.asarray(center)
        self.radius = radius
    
    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def bounding_box(self) -> "Box":
        box = Box(None)
        box.set_center(self.center, self.radius * 2)
        return box

    def contains(self, points, size=0.0):
        center = self.center
        radius = self.radius
        half_size = np.asarray(size) / 2

        return np.linalg.norm(points - center, axis=-1) <= radius - half_size
    
    def contains_data(self, data, cell: bool=False, boxsize: float=1.0):
        center = self.center
        radius = self.radius

        half_size = get_cell_size(data, boxsize=boxsize) / 2 if cell else 0.0

        # TODO: need more accurate intersection
        dist2 = np.zeros(len(data), dtype=float)
        for i, key in enumerate(get_dim_keys()):
            dist2 += (data[key] - center[i])**2
        mask = np.sqrt(dist2) <= radius - half_size
        return mask


class Spheroid(Region):
    def __init__(self, center, radii: np.ndarray):
        self._center = np.asarray(center)
        self.radii = np.asarray(radii)

    @property
    def center(self) -> np.ndarray:
        return self._center

    @property
    def bounding_box(self) -> "Box":
        box = Box(None)
        box.set_center(self.center, self.radii * 2)
        return box

    def contains(self, points, size=0):
        center = self.center
        radii = self.radii
        normed = (points - center) / (radii - size)
        dist = np.linalg.norm(normed, axis=-1)
        mask = dist <= 1
        return mask
    
    def contains_data(self, data, cell: bool=False, boxsize: float=1.0):
        center = self.center
        radii = self.radii
        
        half_size = get_cell_size(data, boxsize=boxsize) / 2 if cell else 0.0

        # TODO: need more accurate intersection
        dist2 = np.zeros(len(data), dtype=float)
        for i, key in enumerate(get_dim_keys()):
            normed = (data[key] - center[i]) / (radii[i] - half_size)
            dist2 += normed**2
        mask = dist2 <= 1
        return mask