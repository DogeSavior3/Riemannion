from pathlib import Path

import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate, to_absolute_path
from tqdm import tqdm

from src.utils.plotting import apply_plot_style


def plot_logs(logs, show=True, save_path=None, save_kwargs=None):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=plt.rcParams["figure.figsize"])

    for label, rel_errors, grad_norms in logs:
        axes[0].plot(rel_errors, label=label)
        axes[1].plot(grad_norms, label=label)
    axes[0].grid()
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$\frac{|\lambda-\lambda^*|}{|\lambda^*|}$")
    axes[0].set_title("Relative error")

    axes[1].grid()
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\|G\|_F$")
    axes[1].set_title("Gradient norm")

    axes[0].legend()
    axes[1].legend()
    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def run_experiment(objective, optimizer, X, true_value, max_iter, tol, desc=""):
    rel_errors = []
    grad_norms = []

    for _ in tqdm(range(max_iter), desc=desc, leave=True):
        optimizer.zero_grad()

        value = objective(X)
        value.backward()

        optimizer.step()

        grad_norm = optimizer.last_grad_norm
        rel_error = abs(value.item() - true_value) / abs(true_value)

        rel_errors.append(rel_error)
        grad_norms.append(grad_norm)

        if grad_norm < tol:
            break

    return rel_errors, grad_norms


def run_rayleigh(cfg):
    settings = cfg.experiments
    torch.manual_seed(settings.seed)
    torch.set_default_dtype(getattr(torch, settings.dtype))
    apply_plot_style(cfg.plotting.style)

    n = settings.n
    K_n = (
        torch.eye(n, n) * 2.0
        + torch.diag(torch.ones(n - 1) * -1.0, 1)
        + torch.diag(torch.ones(n - 1) * -1.0, -1)
    )
    k = torch.arange(1, n + 1)
    eigvals = 2 - 2 * torch.cos(torch.pi * k / (n + 1))
    step_kwargs = {"A": K_n} if "A" in list(cfg.step_rules) else {}
    step_rule = instantiate(cfg.step_rules, **step_kwargs)

    for p in settings.p_list:
        Q, _ = torch.linalg.qr(torch.randn(n, p))
        true_val = torch.mean(eigvals[:p]).item()
        objective = instantiate(cfg.objectives, A=K_n)
        methods = cfg.get("comparison", {settings.label: cfg.retractions})
        logs = []
        for name, retraction_cfg in methods.items():
            X = Q.clone().detach().requires_grad_(True)
            optimizer = instantiate(
                cfg.optimizer,
                params=[X],
                objective=objective,
                step_rule=step_rule,
                retraction=instantiate(retraction_cfg),
            )
            rel_errors, grad_norms = run_experiment(
                objective,
                optimizer,
                X,
                true_val,
                settings.max_iter,
                settings.tol,
                desc=f"p={p} | {name}",
            )
            logs.append((name, rel_errors, grad_norms))
        if settings.show or settings.save:
            save_path = None
            if settings.save:
                method = (
                    "ablation"
                    if "comparison" in cfg
                    else cfg.retractions._target_.split(".")[-2]
                )
                filename = (
                    f"rayleigh_n{n}_p{p}_{method}.{cfg.plotting.save_kwargs.format}"
                )
                save_path = Path(to_absolute_path(settings.output_dir)) / filename
            plot_logs(
                logs,
                show=settings.show,
                save_path=save_path,
                save_kwargs=cfg.plotting.save_kwargs,
            )
