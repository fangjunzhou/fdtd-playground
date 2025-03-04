import logging
from typing import Tuple
import taichi as ti

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ti.dataclass
class Cell2D:
    p: ti.f32
    v: ti.math.vec2

@ti.data_oriented
class Scene2D:
    # Staggered grid for pressure and velocity.
    grid: ti.StructField
    # Simulation grid size.
    size: Tuple[int, int]

    def __init__(self, size: Tuple[int, int]) -> None:
        self.size = size
        self.grid = Cell2D.field(shape=size)


def main():
    # Initialize taichi.
    ti.init(arch=ti.gpu)
    # Construct scene.
    scene = Scene2D((512, 512))
    logger.info(f"Scene grid shape: {scene.grid.shape}.")

if __name__ == "__main__":
    main()
