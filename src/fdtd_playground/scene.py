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


    def apply_velocity(self):
        @ti.kernel
        def clear_mask_velocity():
            for i, j in self.grid.v_grid:
                if self.grid.alpha_grid[i, j] == 1:
                    self.grid.vx_grid[i, j] = 0
                    self.grid.vx_grid[i+1, j] = 0
                    self.grid.vy_grid[i, j] = 0
                    self.grid.vy_grid[i, j+1] = 0
        @ti.kernel
        def apply_velocity():
            for i, j in self.grid.v_grid:
                # TODO: Implement WaveBlender
                if self.grid.alpha_grid[i, j] == 1:
                    self.grid.vx_grid[i, j] += self.grid.v_grid[i, j].x
                    self.grid.vx_grid[i+1, j] += self.grid.v_grid[i, j].x
                    self.grid.vy_grid[i, j] += self.grid.v_grid[i, j].y
                    self.grid.vy_grid[i, j+1] += self.grid.v_grid[i, j].y

        clear_mask_velocity()
        apply_velocity()


    def update_velocity(self):
        @ti.kernel
        def update_velocity():
            dx = self.grid.dx
            dt = self.grid.dt
            sx, sy = self.grid.size
            for i, j in ti.ndrange((1, sx), (0, sy)):
                self.grid.vx_grid[i, j] -= (self.grid.p_grid[i, j] - self.grid.p_grid[i-1, j])/dx * dt
            for i, j in ti.ndrange((0, sx), (1, sy)):
                self.grid.vy_grid[i, j] -= (self.grid.p_grid[i, j] - self.grid.p_grid[i, j-1])/dx * dt

        update_velocity()

    def update_pressure(self):
        @ti.kernel
        def update_pressure():
            dt = self.grid.dt
            c = self.grid.c
            sx, sy = self.grid.size
            for i, j in ti.ndrange((0, sx), (0, sy)):
                dx = self.grid.vx_grid[i+1, j] - self.grid.vx_grid[i, j]
                dy = self.grid.vy_grid[i, j+1] - self.grid.vy_grid[i, j]
                self.grid.p_grid[i, j] -= c**2 * (dx + dy) * dt
        update_pressure()


    def step(self):
        # Rasterize geometries.
        self.rasterize(self.t)
        # Update velocity.
        self.update_velocity()
        # Apply boundary velocity.
        self.apply_velocity()
        # Update pressure.
        self.update_pressure()
        self.t += self.grid.dt



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

    SIZE = (512, 512)
    grid = Grid2D(SIZE, 1/512, 340)
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
    scene.objects.append(circle)

    disp_buf = ti.field(ti.math.vec3, shape=SIZE)

    @ti.kernel
    def clear_disp_buf():
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(0)

    @ti.kernel
    def render_alpha():
        """Render pressure field."""
        for i, j in disp_buf:
            disp_buf[i, j] = ti.math.vec3(grid.alpha_grid[i, j])

    @ti.kernel
    def render_velocity():
        """Render pressure field."""
        for i, j in disp_buf:
            x = grid.vx_grid[i, j]
            y = grid.vy_grid[i, j]
            disp_buf[i, j] = ti.math.vec3((x+1)/2, (y+1)/2, 0)

    gui = ti.GUI("FDTD Simulation", res=SIZE) # pyright: ignore
    frame = 0
    while gui.running:
        clear_disp_buf()
        scene.rasterize(frame * grid.dt)
        scene.apply_velocity()
        render_velocity()
        gui.set_image(disp_buf)
        gui.text(content=f"frame={frame}, t={frame * grid.dt}", pos=[0, 0])
        gui.show()
        frame += 1

if __name__ == "__main__":
    main()
