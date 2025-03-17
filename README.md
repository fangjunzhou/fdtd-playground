# FDTD Playground

A simple FDTD playground

## Run the Simulation

1. Install [uv](https://docs.astral.sh/uv/) for python package management.
2. Install all the required dependencies with `uv sync`.
3. Run the simulation with `uv run start -i <input/audio/path> -o <output/dir/path>`.

Read `uv run start --help` for details about command line argument.

## Example Artifact

Example simulation artifact simulated with

```bash
uv run start -i resources/trumpet_2_low_mono.wav -o resources/result -r -t 10 -b 1 --checkpoint 8192
```

is saved in `resources/result/blend-1-10s`.
