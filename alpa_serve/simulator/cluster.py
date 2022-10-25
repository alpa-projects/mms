"""
The cluster and device mesh abstraction.

This file simulates `alpa/device_mesh.py`.
"""

from itertools import count
from typing import Sequence, Tuple

import numpy as np


class GPU:
    idx = count()

    def __init__(self):
        self.stream_name = next(GPU.idx)


class Mesh:
    def __init__(self, shape: Tuple[int]):
        self.gpus = []
        for i in range(shape[0]):
            for j in range(shape[1]):
                self.gpus.append(GPU())


class MeshGroup:
    def __init__(self, mesh_shapes: Sequence[Tuple[int]]):
        self.meshes = []

        for shape in mesh_shapes:
            self.meshes.append(Mesh(shape))


class VirtualMesh:
    def __init__(self, shape):
        self.shape = shape

        self.submesh_shapes = None
        self.launched_mesh_group = None

    def launch_mesh_group(self, submesh_shapes: Sequence[Tuple[int]]):
        assert self.launched_mesh_group is None

        assert np.prod(self.shape) == sum(np.prod(x) for x in submesh_shapes)

        self.submesh_shapes = tuple(submesh_shapes)
        self.launched_mesh_group = MeshGroup(submesh_shapes)

        return self.launched_mesh_group
