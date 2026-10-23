# Riemannion source layout

This is a scaffold for gradually migrating the existing research scripts.
The packages are intentionally empty; existing scripts continue to live in their original locations.

| Package | Responsibility |
| --- | --- |
| `optimizer/` | Riemannian optimizer and momentum updates. |
| `retractions/` | SVD, QR, polar, geodesic and iterative retractions. |
| `transport/` | Vector transport between tangent spaces. |
| `step_rules/` | Fixed steps, Armijo search and objective-specific step rules. |
| `orthogonalization/` | CANS, Newton–Schulz, Polar Express and polynomial coefficients. |
| `objectives/` | Optimization problems, including the Rayleigh objective. |
| `experiments/` | Shared experiment runners and parameter sweeps. |
| `models/` | Neural network architectures for learning experiments. |
| `datasets/` | Dataset preparation and data loaders. |
| `utils/` | Shared logging, reproducibility and plotting helpers. |

## Configuration

`configs/example.yaml` composes the empty `example.yaml` files in each configuration group.
Add parameters and Hydra `_target_` values when the corresponding implementation is migrated.
No training entry point or existing script is connected to these configurations yet.

Dependency name: `hydra-core`; Python import: `hydra`.
