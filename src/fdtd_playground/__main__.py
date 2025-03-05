from typing import Tuple
import taichi as ti
import argparse

from fdtd_playground.grid import Grid2D

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
        default=(128, 128)
    )
    parser.add_argument(
        "-c",
        "--cell",
        help="Simulation cell size.",
        type=float,
        default=0.01
    )
    parser.add_argument(
        "-t",
        "--time",
        help="Simulation length.",
        type=float,
        default=8
    )
    args = parser.parse_args()

    # Simulation parameters.
    grid_size: Tuple[int, int] = args.grid
    cell_size: float = args.cell
    time: float = args.time

    # Simulation setup.
    scene = Grid2D(grid_size, cell_size)

    disp_buf = ti.field(ti.math.vec3, shape=grid_size)
    @ti.kernel
    def render_p_field():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(scene.p_grid[i, j])

    gui = ti.GUI("FDTD Simulation", res=grid_size) # pyright: ignore
    while gui.running:
        render_p_field()
        gui.set_image(disp_buf)
        gui.show()

if __name__ == "__main__":
    main()
