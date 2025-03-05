from typing import Tuple
import taichi as ti
import argparse

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
        default=(512, 512)
    )
    parser.add_argument(
        "-c",
        "--cell",
        help="Simulation cell size.",
        type=float,
        default=0.01
    )
    args = parser.parse_args()

    # Simulation size.
    grid_size: Tuple[int, int] = args.grid
    cell_size: float = args.cell
    disp_buf = ti.field(ti.math.vec3, shape=grid_size)
    scene = Scene2D(grid_size, cell_size)

    @ti.kernel
    def render_p_field():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(scene.grid[i, j].p)

    gui = ti.GUI("FDTD Simulation", res=grid_size) # pyright: ignore
    while gui.running:
        render_p_field()
        gui.set_image(disp_buf)
        gui.show()

if __name__ == "__main__":
    main()
