# Riemannion

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This project studies optimization on the Stiefel manifold and the adaptation of Muon to the Riemannian setting. We implement Riemannian gradient descent and adapt Riemannion to the Stiefel manifold, comparing them on eigenvalue problems motivated by physical systems and neural network training. It was developed as a third-year coursework project in Applied Mathematics and Information Science (AMI), Faculty of Computer Science (FCS), HSE University.

The repository includes Rayleigh quotient minimization experiments for comparing retractions and a CIFAR-10 classification experiment with a small MLP for comparing Riemannion, Manifold Muon, and optionally AdamW. Experiments are configured with Hydra.

## Installation

The project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

1. Install uv using the [official instructions](https://docs.astral.sh/uv/getting-started/installation/). For example, on Linux or macOS:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Restart your terminal if `uv` is not yet available on your `PATH`.

   On macOS, you can also install it with Homebrew:

   ```bash
   brew install uv
   ```

2. Clone the repository and install the locked dependencies:

   ```bash
   git clone https://github.com/DogeSavior3/RiemannianStiefel.git
   cd RiemannianStiefel
   uv sync --locked --python 3.13
   ```

   uv creates a `.venv` environment and downloads Python 3.13 if needed. Dependencies are defined in `pyproject.toml` and pinned in `uv.lock`.

## How To Use

Run Rayleigh quotient minimization:

```bash
uv run python run_rayleigh.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Run the CIFAR-10 classification experiment:

```bash
uv run python run_cifar.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from [src/configs](src/configs) and `HYDRA_CONFIG_ARGUMENTS` are optional overrides. Default configs are `rayleigh` and `cifar`, respectively. Use `rayleigh_ablation` to compare multiple retractions on the same graph.

Set `experiments.save=true` to save plots and `experiments.show=false` to disable display. Outputs are stored in `output/`.

The default plot style requires a local LaTeX installation. To render without LaTeX, append `'plotting.style={text.usetex:false}'` to the command.

See [demo.ipynb](demo.ipynb) for examples of running experiments and evaluating their results.

## Credits

- [CANS](https://arxiv.org/abs/2506.10935) and [Polar Express](https://arxiv.org/abs/2505.16932), used as retraction methods.
- [Bernstein's Modular Manifolds](https://thinkingmachines.ai/blog/modular-manifolds/), the source of the Manifold Muon implementation used in the CIFAR-10 comparison.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
