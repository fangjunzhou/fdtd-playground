import pathlib
from typing import Tuple
import taichi as ti
import jax.numpy as jnp
import argparse

from fdtd_playground.grid import Grid2D
from fdtd_playground.object import Circle
from fdtd_playground.scene import Scene2D

def main():
    ti.init(arch=ti.gpu)

    parser = argparse.ArgumentParser(
        "fdtd-playground",
        description="A simple FDTD simulation implemented in Taichi."
    )
    parser.add_argument(
        "-g",
        "--grid",
        help="Simulation grid size.",
        type=lambda s: tuple(map(int, s.split(','))),
        default=(256, 256)
    )
    parser.add_argument(
        "-c",
        "--cell",
        help="Simulation cell size.",
        type=float,
        default=5/256
    )
    parser.add_argument(
        "-t",
        "--time",
        help="Simulation length.",
        type=float,
        default=8
    )
    parser.add_argument(
        "-o",
        "--open",
        help="Audio wav file.",
        type=pathlib.Path,
    )
    args = parser.parse_args()

    # Simulation parameters.
    grid_size: Tuple[int, int] = args.grid
    cell_size: float = args.cell
    time: float = args.time
    audio_path: pathlib.Path = args.open

    # Simulation setup.
    grid = Grid2D(grid_size, cell_size)
    scene = Scene2D(grid)

    # Add audio source.
    key_frames = jnp.array([0, 1], dtype=jnp.float32)
    center = jnp.array([
        [2.5, 2],
        [2.5, 3]
    ], dtype=jnp.float32)
    radius = jnp.array([
        0.5,
        1
    ], dtype=jnp.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    circle.load_audio_file(audio_path)
    scene.objects.append(circle)

    # Display.
    disp_buf = ti.field(ti.math.vec3, shape=grid_size)

    @ti.kernel
    def render():
        # Render pressure field
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(grid.p_grid[i, j]/10)
        # Render velocity field
        for i, j in disp_buf:
            if grid.alpha_grid[i, j] == 1:
                v = grid.v_grid[i, j]
                disp_buf[i, j] = ti.math.vec3((v.x + 1)/2, (v.y + 1)/2, 0)

    gui = ti.GUI("FDTD Simulation", res=grid_size) # pyright: ignore
    while gui.running:
        if scene.t < time:
            scene.step()
        render()
        gui.set_image(disp_buf)
        gui.show()

if __name__ == "__main__":
    main()
