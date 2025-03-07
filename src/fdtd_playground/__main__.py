import logging
import pathlib
from typing import Tuple
import taichi as ti
import jax.numpy as jnp
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from scipy.io.wavfile import write

from fdtd_playground.grid import Grid2D
from fdtd_playground.object import BoxObstacle, Circle
from fdtd_playground.scene import Scene2D


logger = logging.getLogger(__name__)


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
        default=(64, 64)
    )
    parser.add_argument(
        "-c",
        "--cell",
        help="Simulation cell size.",
        type=float,
        default=1/64
    )
    parser.add_argument(
        "-b",
        "--blend",
        help="Blend distance in unit of dx.",
        type=float,
        default=1
    )
    parser.add_argument(
        "-t",
        "--time",
        help="Simulation length.",
        type=float,
        default=1
    )
    parser.add_argument(
        "-i",
        "--input",
        help="Audio wav file.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory.",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-r",
        "--render",
        help="Option to render animation.",
        action="store_true",
    )
    parser.add_argument(
        "--anim",
        help="Animation time step size.",
        type=float,
        default=1/60
    )
    parser.add_argument(
        "--fps",
        help="Animation frame rate.",
        type=float,
        default=60
    )
    args = parser.parse_args()

    # Simulation parameters.
    grid_size: Tuple[int, int] = args.grid
    cell_size: float = args.cell
    blend_dist: float = args.blend
    time: float = args.time
    audio_path: pathlib.Path = args.input
    out_path: pathlib.Path = args.output
    render: bool = args.render
    anim_dt: float = args.anim
    fps: int = args.fps

    # Simulation setup.
    grid = Grid2D(grid_size, cell_size)
    scene = Scene2D(grid, 0.0001, 8, 0.05, blend_dist)

    # Add audio source.
    key_frames = jnp.array([0, 0.02], dtype=jnp.float32)
    center = jnp.array([
        [0.75, 0.4],
        [0.75, 0.6]
    ], dtype=jnp.float32)
    radius = jnp.array([
        0.05,
        0.05
    ], dtype=jnp.float32)
    circle = Circle(key_frames, center, radius)
    # Test audio.
    sample_rate = 44100
    audio_length = 8
    samples_t = jnp.linspace(0, audio_length, int(sample_rate * audio_length))
    samples = jnp.sin(440 * 2 * jnp.pi * samples_t)
    # circle.load_audio_file(audio_path)
    circle.load_audio_sample(sample_rate, samples)
    scene.objects.append(circle)

    # Add obstacle.
    # key_frames = jnp.array([0], dtype=jnp.float32)
    # center = jnp.array([
    #     [0.75, 0.4],
    # ], dtype=jnp.float32)
    # radius = jnp.array([
    #     0.05,
    # ], dtype=jnp.float32)
    # circle = Circle(key_frames, center, radius)
    # scene.objects.append(circle)

    key_frames = jnp.array([0, 0.02], dtype=jnp.float32)
    center = jnp.array([
        [0.5, 0.5],
        [0.5, 0.5],
    ], dtype=jnp.float32)
    size = jnp.array([
        [0.01, 0.1],
        [0.01, 0.1],
    ], dtype=jnp.float32)
    rotation = jnp.array([
        -np.pi,
        np.pi
    ], dtype=jnp.float32)
    box = BoxObstacle(key_frames, center, size, rotation)
    scene.objects.append(box)

    # Record samples.
    rx, ry = (0.25, 0.5)
    rx, ry = int(rx / grid.dx), int(ry / grid.dx)
    num_frames = int(time / grid.dt)
    vel_samples = jnp.zeros((num_frames))

    fig, ax = plt.subplots(figsize=(8, 4))
    artists = []
    t = anim_dt
    def draw(frame: int, t: float, dt: float):
        if t >= anim_dt:
            vf = grid.v_grid.to_numpy()
            vf = np.linalg.norm(vf, axis=-1)
            af = grid.alpha_grid.to_numpy()
            pf = grid.p_grid.to_numpy()
            # Plot pressure field.
            img_p = ax.imshow(pf.T, vmin=-10, vmax=10, cmap="coolwarm")
            # Plot normal velocity.
            img_v = ax.imshow(vf.T, alpha=af.T, vmin=-1, vmax=1)
            ax.invert_yaxis()
            patches = [
                mpatches.Patch(label=f"Frame {frame}"),
                mpatches.Patch(label=f"Time {frame * dt:.6f}s")
            ]
            lgds = ax.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
            ax.add_artist(lgds)
            artists.append([img_p, img_v, lgds])
            t = 0
        t += dt
        return t

    for frame in tqdm(range(num_frames)):
        scene.step()
        # Record sample.
        vel_samples = vel_samples.at[frame].set(grid.vx_grid[rx, ry])
        if render:
            t = draw(frame, t, grid.dt)

    logger.info("Saving animation")
    if render:
        anim = animation.ArtistAnimation(fig=fig, artists=artists, interval=int(1000/fps))
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(filename=out_path/"fdtd.mp4", writer=writer)
    logger.info("Animation saved")

    logger.info("Saving audio")
    # Resample audio wave.
    sample_rate = 44100
    vel_samples = jnp.interp(jnp.linspace(0, num_frames, int(sample_rate * time)), jnp.arange(0, num_frames), vel_samples)
    # Finite difference velocity samples to get audio samples.
    sample = (vel_samples[1:] - vel_samples[:-1]) * sample_rate
    sample /= jnp.max(jnp.abs(sample))
    fig, ax = plt.subplots()
    ax.plot(jnp.linspace(0, time, sample.size), sample)
    ax.set_title("Recorded Waveform")
    fig.savefig(out_path/"fdtd-wav.png")
    write(out_path/"record.wav", sample_rate, np.array(sample))
    logger.info("Animation saved")

if __name__ == "__main__":
    main()
