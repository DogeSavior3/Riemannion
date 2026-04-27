import time
from collections import deque
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from hyperspherical_descent import hyperspherical_descent
from manifold_muon import manifold_muon
from torch.optim import AdamW
from torch.utils.data import DataLoader

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from optimizer import RiemannianOptimizerSt
from retraction import (
    svd_retraction,
    polar_retraction,
    qr_retraction,
    make_cans_retraction,
    make_newton_schulz_retraction,
    make_polar_express_retraction,
)
from step_algorithm import fixed_step_rule

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. EPS-SPECIFIC SETTINGS
plt.rcParams["savefig.format"] = "eps"
plt.rcParams["ps.useafm"] = True
plt.rcParams["pdf.use14corefonts"] = True
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["pdf.fonttype"] = 42

# 2. FIGURE SIZE
width_in = 6
height_in = width_in * 0.5
plt.rcParams["figure.figsize"] = (width_in, height_in)

# 3. FONT CONFIGURATION
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"

# 4. LINE QUALITY
plt.rcParams["lines.linewidth"] = 1.2
plt.rcParams["lines.markersize"] = 6

# 5. FONT SIZES
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["xtick.labelsize"] = 9
plt.rcParams["ytick.labelsize"] = 9
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["axes.titlesize"] = 10
plt.rcParams["lines.markersize"] = 6
plt.rcParams["lines.markeredgewidth"] = 1.0

plt.rcParams["legend.handlelength"] = 1.5
plt.rcParams["legend.handletextpad"] = 0.4
plt.rcParams["legend.borderaxespad"] = 0.2
plt.rcParams["axes.titlepad"] = 4

save_kwargs = {
    "format": "eps",
    "dpi": 300,
    "bbox_inches": "tight",
    "transparent": True,
}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.49139968, 0.48215827, 0.44653124),
        (0.24703233, 0.24348505, 0.26158768)
    )
])

train_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=True, transform=transform, download=True
)
test_dataset = torchvision.datasets.CIFAR10(
    root="./data", train=False, transform=transform, download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=1024, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=1024, shuffle=False)


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
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(epochs, initial_lr, update, wd, smooth_window=20):
    model = MLP().to(device)
    criterion = nn.CrossEntropyLoss()

    if update == AdamW:
        optimizer = AdamW(model.parameters(), lr=initial_lr, weight_decay=wd)
    else:
        assert update in [manifold_muon, hyperspherical_descent]
        optimizer = None

    steps = epochs * len(train_loader)
    step = 0

    if optimizer is None:
        for p in model.parameters():
            p.data = update(p.data, torch.zeros_like(p.data), eta=0)

    epoch_losses = []
    test_accs = []
    train_accs = []

    time_history = []
    loss_history = []
    recent_losses = deque(maxlen=smooth_window)

    start_total = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

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

            recent_losses.append(loss.item())
            smooth_loss = sum(recent_losses) / len(recent_losses)

            elapsed_now = time.time() - start_total
            time_history.append(elapsed_now)
            loss_history.append(smooth_loss)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                smooth=f"{smooth_loss:.4f}",
                lr=f"{lr:.5f}",
                t=f"{elapsed_now:.1f}s",
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        test_acc, train_acc = eval_model(model)
        test_accs.append(test_acc)
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    elapsed = time.time() - start_total

    return model, epoch_losses, test_accs, train_accs, elapsed, time_history, loss_history


def train_riemannian_muon(epochs, initial_lr, momentum, retraction, smooth_window=20):
    model = MLP().to(device)

    for p in model.parameters():
        p.data.copy_(stiefel_init_like(p.data))

    criterion = nn.CrossEntropyLoss()

    optimizer = RiemannianOptimizerSt(
        model.parameters(),
        lr=initial_lr,
        momentum=momentum,
        retraction=retraction,
        step_rule=fixed_step_rule,
        objective=None,
        muon=True,
    )

    steps = epochs * len(train_loader)
    step = 0

    epoch_losses = []
    test_accs = []
    train_accs = []

    time_history = []
    loss_history = []
    recent_losses = deque(maxlen=smooth_window)

    start_total = time.time()

    for epoch in range(epochs):
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=True)

        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            lr = initial_lr * (1 - step / steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            optimizer.step()

            step += 1
            running_loss += loss.item()

            avg_loss = running_loss / (i + 1)

            recent_losses.append(loss.item())
            smooth_loss = sum(recent_losses) / len(recent_losses)

            elapsed_now = time.time() - start_total
            time_history.append(elapsed_now)
            loss_history.append(smooth_loss)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                avg=f"{avg_loss:.4f}",
                smooth=f"{smooth_loss:.4f}",
                lr=f"{lr:.5f}",
                t=f"{elapsed_now:.1f}s",
            )

        epoch_loss = running_loss / len(train_loader)
        epoch_losses.append(epoch_loss)

        test_acc, train_acc = eval_model(model)
        test_accs.append(test_acc)
        train_accs.append(train_acc)

        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    elapsed = time.time() - start_total

    return model, epoch_losses, test_accs, train_accs, elapsed, time_history, loss_history


def eval_model(model):
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
    model.train()
    return accs[0], accs[1]


def plot_results(results_list):
    plt.figure(figsize=(width_in, height_in))

    plt.subplot(1, 2, 1)
    for res in results_list:
        epochs = range(1, len(res["epoch_losses"]) + 1)
        plt.plot(epochs, res["epoch_losses"], label=f"{res['method']} ({res['elapsed']:.2f}s)")
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for res in results_list:
        epochs = range(1, len(res["test_accs"]) + 1)
        plt.plot(epochs, res["test_accs"], label=f"{res['method']} ({res['elapsed']:.2f}s)")
    plt.xlabel("Epoch")
    plt.ylabel("Test accuracy $(\\%)$")
    plt.title("Test accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("exp_nn.eps", **save_kwargs)
    plt.savefig("exp_nn.png", dpi=200, bbox_inches="tight")
    plt.show()


def plot_results_time_loss(results_list, max_time=None):
    plt.figure(figsize=(width_in * 0.7, height_in * 0.9))

    for res in results_list:
        times = res["time_history"]
        losses = res["loss_history"]

        if max_time is not None:
            filtered = [(t, l) for t, l in zip(times, losses) if t <= max_time]
            if len(filtered) == 0:
                continue
            times, losses = zip(*filtered)

        plt.plot(times, losses, label=f"{res['method']} ({res['elapsed']:.2f}s)")

    plt.xlabel("Time (s)")
    plt.ylabel("Smoothed train loss")
    plt.title("Loss vs time" + (f" (first {max_time}s)" if max_time is not None else ""))
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    suffix = f"_{int(max_time)}s" if max_time is not None else ""
    plt.savefig(f"exp_nn_time_loss{suffix}.eps", **save_kwargs)
    plt.savefig(f"exp_nn_time_loss{suffix}.png", dpi=200, bbox_inches="tight")
    plt.show()

results_all = []

retractions_to_test = [
    ("SVD", svd_retraction),
    ("QR", qr_retraction),
    ("Polar", polar_retraction),
    ("CANS", make_cans_retraction()),
    ("Newton-Schulz", make_newton_schulz_retraction()),
    ("Polar Express", make_polar_express_retraction())
]

for name, retraction in retractions_to_test:
    print(f"Training with: riemannion + {name}")

    model, epoch_losses, test_accs, train_accs, elapsed, time_history, loss_history = train_riemannian_muon(
        epochs=6,
        initial_lr=0.08351431410962708,
        momentum=0.1,
        smooth_window=7,
        retraction=retraction,
    )

    results_all.append({
        "method": f"riemannion-{name}",
        "epoch_losses": epoch_losses,
        "test_accs": test_accs,
        "train_accs": train_accs,
        "elapsed": elapsed,
        "time_history": time_history,
        "loss_history": loss_history,
    })

plot_results(results_all)
plot_results_time_loss(results_all)
plot_results_time_loss(results_all, max_time=100)