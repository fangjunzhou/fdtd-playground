from typing import List
import taichi as ti

from fdtd_playground.grid import Grid2D
from fdtd_playground.object import Object


class Scene2D:
    # Scene grid.
    grid: Grid2D
    # Scene objects.
    objects: List[Object]

    # Simulation states.
    t: ti.f32 = 0
