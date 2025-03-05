import logging
from typing import Tuple
import taichi as ti

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ti.dataclass
class Cell2D:
    # Cell pressure.
    p: ti.f32
    # Staggered velocity
    v: ti.math.vec2

@ti.data_oriented
class Scene2D:
    # Staggered grid for pressure and velocity.
    grid: ti.StructField
    # Simulation grid size.
    size: Tuple[int, int]
    # Simulation cell size.
    dx: ti.f32
    # Sound wave speed.
    c: ti.f32
    # Time step size.
    dt: ti.f32

    def __init__(self,
                 size: Tuple[int, int],
                 dx: ti.f32,
                 c: ti.f32 = 340) -> None:
        self.size = size
        self.dx = dx
        self.c = c
        self.dt = dx / (ti.sqrt(2) * c)

        self.grid = Cell2D.field(shape=size)


def main():
    # Initialize taichi.
    ti.init(arch=ti.gpu)
    # Construct scene.
    scene = Scene2D((512, 512), 0.01)
    logger.info(f"Scene grid shape: {scene.grid.shape}.")

if __name__ == "__main__":
    main()
