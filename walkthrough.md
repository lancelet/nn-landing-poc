# Quadcopter Optimal Landing: Neural Network Controller

## Introduction

Let's train a neural network to perform energy-optimal quadcopter landings!
This is my re-implementation of a research paper from the European Space 
Agency. If you want to follow along, best grab a copy:

- Sánchez-Sánchez et al. (2016)
  [Optimal Real-Time Landing Using Deep Networks](https://www.esa.int/gsp/ACT/doc/AI/pub/ACT-RPR-AI-2016-ICATT-optimal_landing_deep_networks.pdf)
  Proceedings of the 6th International Conference on Astrodynamics Tools
  and Techniques, ICATT, Vol. 12, European Space Agency, The Netherlands,
  Aug. 2016, pp. 2493-2537.
  
This project implements the model from section 3.1 of the paper: "Pinpoint 
landing (multicopter model)".

## Training Data

Training data is created using the
[`dymos`](https://openmdao.github.io/dymos/) library in Python. I'm using
the [`uv`](https://github.com/astral-sh/uv) project management tool:

```bash
cd generate_data
uv run quadcopter.py
```

It will spawn multiple threads to generate trajectories contained in `npz`
files under the `trajectories/quadcopter` directory. It takes about 50 minutes 
on an M1 MacBook Pro. It is configured to generate 15,000 example trajectories.

It works like this:

1. For each trajectory sample, we pick a random starting state inside a 
   training domain.
2. We run `dymos` to solve for an energy-optimal trajectory given that 
   starting state, using Gauss-Lobatto 
   [collocation](https://en.wikipedia.org/wiki/Collocation_method).
3. Each optimized trajectory is simulated by integrating the equations of 
   motion forward in time, and then dumped to an `npz` file.
   
Once you have generated a few `npz` files, you can take a look at one:

```
$ cd generate_data
$ uv run python3
Python 3.12.5 (main, Aug  6 2024, 19:08:49) [Clang 16.0.6 ] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import numpy as np
>>> data = np.load("../trajectories/quadcopter/0000000.npz")
>>> traj = data["trajectory"]
>>> traj.shape
(91, 8)
```

Each row is a time sample. The 8 columns are (in order): `time`, `x`,
`z`, `theta`, `vx`, `vz`, `thrust` (`u1` in the paper) and `omega` (`u2` in 
the paper). `omega` is the rate of change of the angle of the quadcopter.

## Training

Training is done using the Rust [`burn`](https://burn.dev/) library. You can
use [`rustup`](https://rustup.rs/) to get Rust installed if you're not
familiar with it. To train based on the trajectories we just created, run the
`train` binary. Run using `--release` since that makes training a lot faster.
It takes about 40 minutes on an M1 MacBook Pro:

```bash
cd nn
cargo run --release --bin train
```

During training, we sample the trajectories to produce a corpus of inputs and
outputs:

- Inputs: State of the quadcopter: `x`, `z`, `theta`, `vx`, `vz`.
- Outputs: Control signals: `thrust`, `omega`.

Each chosen sample from the optimized trajectories provides an input-output
example, which is used for supervised training.

The neural network used here is a very basic feed-forward network, with 5
layers. Internally, it uses ReLU activations, and on output it uses a clamped
ReLU that is described in a follow-up paper by Sánchez-Sánchez.

## Inference

Inference is pretty basic for now and geared toward visualization.

You can generate animated frames for a landing using the `vis_results` binary:

```bash
cd nn
cargo run --release --bin vis_results
```

This will dump PNG files in the `./animation` directory. You can convert them
to a movie with `ffmpeg` like this:

```bash
ffmpeg -framerate 60 -i ./animation/%05d.png -vcodec libx264 -s 1080x1080 -pix_fmt yuv420p animation.mov
```
