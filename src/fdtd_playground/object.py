from abc import ABC, abstractmethod
import argparse
import logging
import taichi as ti
import pathlib
import jax.numpy as jnp
from scipy.io.wavfile import read
import matplotlib.pyplot as plt

from fdtd_playground.grid import Grid2D

logger = logging.getLogger(__name__)

class Object(ABC):
    @abstractmethod
    def rasterize_alpha(self, scene: Grid2D, t: float):
        pass

    @abstractmethod
    def rasterize(self, scene: Grid2D, t: float):
        pass

class Circle(Object):
    key_frames: jnp.ndarray
    center: jnp.ndarray
    radius: jnp.ndarray
    sample_rate: int = -1
    samples: jnp.ndarray = jnp.array([])
    normal_v: jnp.ndarray = jnp.array([])

    def __init__(self,
                 key_frames: jnp.ndarray,
                 center: jnp.ndarray,
                 radius: jnp.ndarray) -> None:
        self.key_frames = key_frames
        self.center = center
        self.radius = radius

    def load_audio_sample(self, sample_rate: int, samples: jnp.ndarray):
        self.sample_rate = sample_rate
        self.samples = samples

    def load_audio_file(self, audio_path: pathlib.Path):
        self.sample_rate, self.samples = read(audio_path)

    def integrate_velocity(self, drift_correction: int = 8192, avg_sample: int = 32):
        # Numerical integration.
        self.normal_v = jnp.cumsum(self.samples / self.sample_rate)
        # Solve drift.
        trim = self.normal_v.size % avg_sample
        avg = self.normal_v[:-trim].reshape((-1, avg_sample))
        avg = jnp.mean(avg, axis=-1).flatten()
        low_res_idx = jnp.arange(0, avg.size - 1, int(avg.size / drift_correction))
        low_res = avg[low_res_idx]
        offset = jnp.interp(jnp.linspace(0, low_res.size, self.normal_v.size), jnp.arange(0, low_res.size), low_res)
        self.normal_v -= offset


    def get_center_radius(self, t):
        # Interpolate center and radius.
        center = self.center[-1]
        radius = self.radius[-1]
        if self.key_frames[-1] <= t:
            return center, radius

        for i in range(self.key_frames.size - 1):
            if self.key_frames[i] <= t:
                curr_t, next_t = self.key_frames[i], self.key_frames[i+1]
                alpha = (t - curr_t) / (next_t - curr_t)
                curr_c, next_c = self.center[i], self.center[i+1]
                center = next_c * alpha + curr_c * (1 - alpha)
                curr_r, next_r = self.radius[i], self.radius[i+1]
                radius = next_r * alpha + curr_r * (1 - alpha)
        return center, radius


    def get_normal_v(self, t):
        velocity = 0
        i = int(self.sample_rate * t)
        if i >= 0 and i < self.samples.size - 1:
            alpha = self.sample_rate * t - i
            curr_v, next_v = self.normal_v[i], self.normal_v[i+1]
            velocity = curr_v * alpha + next_v * (1 - alpha)
        return velocity


    def rasterize_alpha(self, scene: Grid2D, t: float):
        c, r = self.get_center_radius(t)

        @ti.kernel
        def rasterize_alpha(c: ti.math.vec2, r: ti.f32):
            for i, j in scene.alpha_grid:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                dist = ti.math.length(pos - c)
                if dist < r:
                    scene.alpha_grid[i, j] = 1

        rasterize_alpha(ti.math.vec2(c), r.item())


    def rasterize(self, scene: Grid2D, t: float):
        c, r = self.get_center_radius(t)
        v = self.get_normal_v(t)
        # Rasterize velocity field.
        sx, sy = scene.size
        v_field = ti.field(ti.math.vec2, shape=(sx, sy))

        @ti.kernel
        def rasterize_velocity(c: ti.math.vec2, r: ti.f32, v: ti.f32):
            for i, j in v_field:
                pos = ti.math.vec2(i * scene.dx, j * scene.dx)
                dist = ti.math.length(pos - c)
                if dist < r:
                    normal = (pos - c) / dist
                    v_field[i, j] = v * normal

        rasterize_velocity(ti.math.vec2(c), r.item(), v.item())
        # TODO: Apply velocity field.


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

    # Test objects.
    key_frames = jnp.array([0, 1], dtype=jnp.float32)
    center = jnp.array([
        [0.5, 0.25],
        [0.5, 0.75]
    ], dtype=jnp.float32)
    radius = jnp.array([
        0.25,
        0.25
    ], dtype=jnp.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    circle.load_audio_file(audio_path)
    circle.integrate_velocity()
    num_samples = circle.samples.size
    audio_length = num_samples / circle.sample_rate
    samples_t = jnp.linspace(0, audio_length, num_samples)
    # plt.plot(samples_t, sphere.samples)
    plt.plot(samples_t, circle.normal_v)
    plt.show()


if __name__ == "__main__":
    main()
