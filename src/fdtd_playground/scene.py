import argparse
import logging
import pathlib
from typing import List
import taichi as ti
import jax.numpy as jnp

from fdtd_playground.grid import Grid2D
from fdtd_playground.object import Object, Circle


logger = logging.getLogger(__name__)


class Scene2D:
    # Scene grid.
    grid: Grid2D
    # Scene objects.
    objects: List[Object]

    # Simulation states.
    t: ti.f32 = 0

    def __init__(self, grid: Grid2D) -> None:
        self.grid = grid
        self.objects = []

    def rasterize(self, t: float):
        @ti.kernel
        def clear_alpha():
            for i, j in self.grid.alpha_grid:
                self.grid.alpha_grid[i, j] = 0

        clear_alpha()
        for obj in self.objects:
            obj.rasterize_alpha(self.grid, t)
            obj.rasterize(self.grid, t)


def main():
    # Initialize taichi.
    ti.init(arch=ti.gpu)

    parser = argparse.ArgumentParser(
        "object",
        description="Audio object."
    )
    parser.add_argument(
        "-o",
        "--open",
        help="Audio wav file.",
        type=pathlib.Path,
    )
    args = parser.parse_args()
    audio_path: pathlib.Path = args.open

    SIZE = (128, 128)
    grid = Grid2D(SIZE, 1/128, 10)
    grid.dt = 1/60
    scene = Scene2D(grid)

    # Test objects.
    key_frames = jnp.array([0, 1], dtype=jnp.float32)
    center = jnp.array([
        [0.5, 0.25],
        [0.5, 0.75]
    ], dtype=jnp.float32)
    radius = jnp.array([
        0.1,
        0.25
    ], dtype=jnp.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    circle.load_audio_file(audio_path)
    circle.integrate_velocity()
    scene.objects.append(circle)

    disp_buf = ti.field(ti.math.vec3, shape=SIZE)

    @ti.kernel
    def render_alpha():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(grid.alpha_grid[i, j])

    gui = ti.GUI("FDTD Simulation", res=SIZE) # pyright: ignore
    frame = 0
    while gui.running:
        scene.rasterize(frame * grid.dt)
        render_alpha()
        gui.set_image(disp_buf)
        gui.text(content=f"frame={frame}, t={frame * grid.dt}", pos=[0, 0])
        gui.show()
        frame += 1

if __name__ == "__main__":
    main()
