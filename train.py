"""Train and test small MNIST classifiers: MLP, CNN, or Transformer encoder.

Usage:
    python train.py --model mlp
    python train.py --model cnn --augment
    python train.py --model transformer --augment --epochs 15
"""
import argparse, json, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------- Models ----------
class MLP(nn.Module):
    """Shallow MLP: one hidden layer."""
    def __init__(self, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(28 * 28, hidden), nn.ReLU(),
            nn.Dropout(0.2), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """Two conv blocks followed by a small classifier."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))  # 7x7
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class TransformerEncoderNet(nn.Module):
    """Split the image into 4x4 patches (49 'words'), run a Transformer encoder."""
    def __init__(self, patch=4, dim=64, heads=4, layers=2, ff=128):
        super().__init__()
        n_patches = (28 // patch) ** 2
        self.embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=ff,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 10)

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)           # (B, 49, dim)
        x = torch.cat([self.cls.expand(x.size(0), -1, -1), x], dim=1) + self.pos
        x = self.norm(self.encoder(x))
        return self.head(x[:, 0])                              # use the CLS token


MODELS = {"mlp": MLP, "cnn": CNN, "transformer": TransformerEncoderNet}


# ---------- Data ----------
def get_loaders(augment, batch_size):
    norm = transforms.Normalize((0.1307,), (0.3081,))
    train_tf = [transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))] if augment else []
    train_tf += [transforms.ToTensor(), norm]
    test_tf = transforms.Compose([transforms.ToTensor(), norm])  # never augment the test set
    train = datasets.MNIST("data", train=True, download=True, transform=transforms.Compose(train_tf))
    test = datasets.MNIST("data", train=False, download=True, transform=test_tf)
    return (DataLoader(train, batch_size=batch_size, shuffle=True),
            DataLoader(test, batch_size=1000))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
    return 100.0 * correct / len(loader.dataset)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=MODELS, required=True)
    p.add_argument("--augment", action="store_true")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    torch.manual_seed(a.seed)
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    train_loader, test_loader = get_loaders(a.augment, a.batch_size)
    model = MODELS[a.model]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss()

    history, start = [], time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_loader:           # TRAIN split only
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        sched.step()
        test_acc = evaluate(model, test_loader, device)   # TEST split, reporting only
        history.append({"epoch": epoch, "train_loss": total / len(train_loader.dataset), "test_acc": test_acc})
        print(f"[{a.model} aug={a.augment}] epoch {epoch}: loss {history[-1]['train_loss']:.4f}  test acc {test_acc:.2f}%")

    result = {
        "model": a.model, "augmentation": a.augment, "epochs": a.epochs,
        "parameters": sum(p.numel() for p in model.parameters()),
        "final_test_accuracy": history[-1]["test_acc"],   # final epoch, not best-on-test
        "train_seconds": round(time.time() - start, 1), "history": history,
    }
    os.makedirs("results", exist_ok=True)
    path = f"results/{a.model}_{'aug' if a.augment else 'noaug'}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print("Saved", path)


if __name__ == "__main__":
    main()
