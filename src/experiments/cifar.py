import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torchvision
from hydra.utils import instantiate, to_absolute_path
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from src.utils.plotting import apply_plot_style


def get_loaders(data_dir, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                (0.49139968, 0.48215827, 0.44653124),
                (0.24703233, 0.24348505, 0.26158768),
            ),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def stiefel_init_like(W):
    should_transpose = W.size(-2) < W.size(-1)
    if should_transpose:
        A = torch.randn(W.size(-1), W.size(-2), dtype=W.dtype, device=W.device)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        return Q.mT.contiguous()
    else:
        A = torch.randn_like(W)
        Q, _ = torch.linalg.qr(A, mode="reduced")
        return Q.contiguous()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(
    model,
    train_loader,
    test_loader,
    device,
    epochs,
    initial_lr,
    optimizer=None,
    update=None,
):
    criterion = nn.CrossEntropyLoss()
    steps = epochs * len(train_loader)
    step = 0

    epoch_losses = []
    test_accs = []
    train_accs = []

    start_total = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            model.zero_grad()
            loss.backward()

            lr = initial_lr * (1 - step / steps)

            with torch.no_grad():
                if optimizer is None:
                    for p in model.parameters():
                        p.data = update(p, p.grad, eta=lr)
                else:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = lr
                    optimizer.step()

            step += 1
            running_loss += loss.item()

            avg_loss = running_loss / (i + 1)
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.5f}"
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        test_acc, train_acc = evaluate(model, train_loader, test_loader, device)
        test_accs.append(test_acc)
        train_accs.append(train_acc)

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss:.4f}")

    elapsed = time.time() - start_total

    return model, epoch_losses, test_accs, train_accs, elapsed


def evaluate(model, train_loader, test_loader, device):
    model.eval()
    with torch.no_grad():
        accs = []
        for dataloader in [test_loader, train_loader]:
            correct = 0
            total = 0
            for images, labels in dataloader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            accs.append(100 * correct / total)

    print(f"Accuracy on test set:  {accs[0]:.2f}%")
    print(f"Accuracy on train set: {accs[1]:.2f}%")
    return accs[0], accs[1]


def plot_results(results_list, show=True, save_path=None, save_kwargs=None):
    fig = plt.figure(figsize=plt.rcParams["figure.figsize"])

    plt.subplot(1, 2, 1)
    for res in results_list:
        epochs = range(1, len(res["epoch_losses"]) + 1)
        plt.plot(
            epochs,
            res["epoch_losses"],
            label=f"{res['method']} ({res['elapsed']:.2f}s)",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for res in results_list:
        epochs = range(1, len(res["test_accs"]) + 1)
        plt.plot(
            epochs, res["test_accs"], label=f"{res['method']} ({res['elapsed']:.2f}s)"
        )
    plt.xlabel("Epoch")
    plt.ylabel(r"Test accuracy $(\%)$")
    plt.title("Test accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, **save_kwargs)
    if show:
        plt.show()
    plt.close(fig)


def run_cifar(cfg):
    settings = cfg.experiments
    torch.manual_seed(settings.seed)
    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if settings.device == "auto"
        else settings.device
    )
    apply_plot_style(cfg.plotting.style)
    train_loader, test_loader = get_loaders(
        to_absolute_path(settings.data_dir),
        settings.batch_size,
    )
    results = []
    for name in settings.methods:
        method = cfg.comparison[name]
        model = MLP().to(device)
        optimizer = None
        update = None
        if method["update"] is not None:
            update = instantiate(method["update"])
            for p in model.parameters():
                p.data.copy_(update(p.data, torch.zeros_like(p.data), eta=0))
        else:
            if method.stiefel_init:
                for p in model.parameters():
                    p.data.copy_(stiefel_init_like(p.data))
            optimizer = instantiate(method.optimizer, params=model.parameters())
        print(f"Training with: {name}")
        model, epoch_losses, test_accs, train_accs, elapsed = train(
            model,
            train_loader,
            test_loader,
            device,
            settings.epochs,
            method.lr,
            optimizer=optimizer,
            update=update,
        )
        results.append(
            {
                "method": name,
                "epoch_losses": epoch_losses,
                "test_accs": test_accs,
                "train_accs": train_accs,
                "elapsed": elapsed,
            }
        )
    save_path = None
    if settings.save:
        save_path = Path(to_absolute_path(settings.output_dir)) / (
            f"cifar_{'_'.join(settings.methods)}.{cfg.plotting.save_kwargs.format}"
        )
    plot_results(
        results,
        show=settings.show,
        save_path=save_path,
        save_kwargs=dict(cfg.plotting.save_kwargs),
    )
    return results
