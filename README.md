# Quadcopter Optimal Landing: Neural Network Controller

![GitHub CI](https://github.com/lancelet/nn-landing-poc/actions/workflows/dry-run.yml/badge.svg)

WARNING: This is an initial proof-of-concept, not a carefully polished 
project!

Watch a video of landings on YouTube:

[![Watch a video on multiple landings on YouTube](https://img.youtube.com/vi/q3VXqRNPi9Q/0.jpg)](https://www.youtube.com/watch?v=q3VXqRNPi9Q)

## Quick Instructions

```bash
# Generate Data
pushd generate_data
uv run quadcopter.py  # Wait 50 minutes
popd

# Train
pushd nn
cargo run --release --bin train  # Wait 40 minutes
popd

# Inference
pushd nn
cargo run --release --bin vis_results  # Pretty quick
# Creates PNG frames from animated trajectory in nn/animation
popd
```

See [here](walkthrough.md) for a detailed walkthrough.

## Slides

There are some slides from a presentation to the Sydney Functional Programming
Group here: 
[https://lancelet.github.io/nn-landing-poc](https://lancelet.github.io/nn-landing-poc).