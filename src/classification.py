"""
Stage 2 – ResNet50 classification: model, dataset, training loop.
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm.auto import tqdm
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
)

from src.constants import CLASS_NAMES
from src.utils import AverageMeter, EarlyStopping, fmt_time




def get_train_transforms(input_size: int = 224, cfg: Optional[dict] = None) -> transforms.Compose:
    cfg = cfg or {}
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomRotation(cfg.get("rotation_degrees", 15)),
        transforms.RandomHorizontalFlip(0.5 if cfg.get("horizontal_flip", True) else 0.0),
        transforms.ColorJitter(
            brightness=cfg.get("brightness", 0.2),
            contrast=cfg.get("contrast", 0.2),
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def get_val_transforms(input_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class BrainTumorDataset(Dataset):
    """Dataset that reads (image_path, label) pairs.

    Non-existent paths are filtered out at construction time so that
    DataLoader workers (num_workers > 0) never hit FileNotFoundError.
    """

    def __init__(
        self,
        samples: List[Tuple[str, int]],
        transform: Optional[transforms.Compose] = None,
    ):
        valid = [(p, l) for p, l in samples if Path(p).exists()]
        if len(valid) < len(samples):
            import warnings
            warnings.warn(
                f"BrainTumorDataset: dropped {len(samples) - len(valid)} "
                f"missing files ({len(valid)} remaining)"
            )
        self.samples = valid
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ------------------------------------------------------------------ #
#  Model
# ------------------------------------------------------------------ #

def build_resnet50(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ------------------------------------------------------------------ #
#  Class-weight computation
# ------------------------------------------------------------------ #

def compute_class_weights(labels: List[int], num_classes: int = 4) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    weights *= num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ------------------------------------------------------------------ #
#  Training & evaluation loops
# ------------------------------------------------------------------ #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_str: str = "",
) -> float:
    model.train()
    loss_meter = AverageMeter()
    correct, total = 0, 0

    pbar = tqdm(loader, desc=f"  Train {epoch_str}", leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", acc=f"{correct/total:.3f}")

    return loss_meter.avg


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "  Val",
) -> Tuple[float, Dict[str, float], np.ndarray]:
    """Return (val_loss, metrics_dict, confusion_matrix)."""
    model.eval()
    loss_meter = AverageMeter()
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=desc, leave=False, unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_meter.update(loss.item(), images.size(0))
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))
    return loss_meter.avg, metrics, cm


def train_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: torch.device,
    epochs: int = 50,
    patience: int = 10,
    save_path: str = "models/classification/best_resnet50.pth",
) -> Dict[str, list]:
    """Full training loop with early stopping. Returns history dict."""
    early_stop = EarlyStopping(patience=patience)
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "accuracy": [], "precision": [], "recall": [], "f1": [],
        "lr": [], "epoch_time": [],
    }
    best_f1 = 0.0
    train_start = time.time()

    print(f"{'='*70}")
    print(f"  Training ResNet50  |  {epochs} epochs  |  patience={patience}  |  device={device}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        epoch_str = f"{epoch}/{epochs}"

        current_lr = optimizer.param_groups[0]["lr"]

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch_str=epoch_str,
        )
        val_loss, metrics, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - train_start
        eta = epoch_time * (epochs - epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        for k in ("accuracy", "precision", "recall", "f1"):
            history[k].append(metrics[k])

        improved = metrics["f1"] > best_f1
        marker = " *" if improved else ""

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f} | "
            f"lr={current_lr:.2e} | "
            f"{fmt_time(epoch_time)}/ep  ETA {fmt_time(eta)}{marker}"
        )

        if improved:
            best_f1 = metrics["f1"]
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (F1={best_f1:.4f})")

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no val_loss improvement for {patience} epochs)")
            break

    total_time = time.time() - train_start
    print(f"{'='*70}")
    print(f"  Training finished in {fmt_time(total_time)}  |  Best F1={best_f1:.4f}")
    print(f"{'='*70}")

    return history


# ------------------------------------------------------------------ #
#  Grid search (simple)
# ------------------------------------------------------------------ #

def grid_search_hyperparams(
    train_samples: List[Tuple[str, int]],
    val_samples: List[Tuple[str, int]],
    lr_candidates: List[float],
    bs_candidates: List[int],
    device: torch.device,
    epochs: int = 15,
    num_classes: int = 4,
    input_size: int = 224,
    aug_cfg: Optional[dict] = None,
) -> Dict:
    """Run a lightweight grid search over LR × batch_size."""
    results = []
    best_f1 = 0.0
    best_params = {}

    labels = [s[1] for s in train_samples]
    class_weights = compute_class_weights(labels, num_classes).to(device)

    combos = [(lr, bs) for lr in lr_candidates for bs in bs_candidates]
    for combo_i, (lr, bs) in enumerate(combos, 1):
        print(f"\n[{combo_i}/{len(combos)}] Grid search: lr={lr}, bs={bs}")
        t0 = time.time()
        train_ds = BrainTumorDataset(train_samples, get_train_transforms(input_size, aug_cfg))
        val_ds = BrainTumorDataset(val_samples, get_val_transforms(input_size))
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2)

        model = build_resnet50(num_classes).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

        for epoch in tqdm(range(1, epochs + 1), desc=f"  lr={lr} bs={bs}", leave=False):
            train_one_epoch(model, train_loader, criterion, optimizer, device)

        _, metrics, _ = evaluate(model, val_loader, criterion, device, desc="  Eval")
        f1 = metrics["f1"]
        elapsed = time.time() - t0
        print(f"  Result: acc={metrics['accuracy']:.4f}, f1={f1:.4f} ({fmt_time(elapsed)})")
        results.append({"lr": lr, "bs": bs, **metrics})

        if f1 > best_f1:
            best_f1 = f1
            best_params = {"lr": lr, "bs": bs}

    print(f"\nBest grid-search params: {best_params} (F1={best_f1:.4f})")
    return {"results": results, "best_params": best_params, "best_f1": best_f1}

# ------------------------------------------------------------------ #
#  Dual-input ResNet50 (cropped patch + original MRI)
# ------------------------------------------------------------------ #

def build_original_lookup(samples: List[Tuple[str, int]]) -> Dict[Tuple[int, str], str]:
    """Map (label, filename) -> original image path."""
    return {(int(label), Path(path).name): str(path) for path, label in samples}


class DualInputBrainTumorDataset(Dataset):
    """Return a 6-channel tensor: [patch RGB (3), original RGB (3)]."""

    def __init__(
        self,
        patch_samples: List[Tuple[str, int]],
        original_lookup: Dict[Tuple[int, str], str],
        patch_transform: Optional[transforms.Compose] = None,
        original_transform: Optional[transforms.Compose] = None,
    ):
        self.samples = [(str(path), int(label)) for path, label in patch_samples if Path(path).exists()]
        self.original_lookup = original_lookup
        self.patch_transform = patch_transform
        self.original_transform = original_transform if original_transform is not None else patch_transform
        self.to_tensor = transforms.ToTensor()
        dropped = len(patch_samples) - len(self.samples)
        if dropped > 0:
            import warnings
            warnings.warn(f"DualInputBrainTumorDataset: dropped {dropped} missing patch files")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        patch_path, label = self.samples[idx]
        patch_img = Image.open(patch_path).convert("RGB")
        original_path = self.original_lookup.get((label, Path(patch_path).name), patch_path)
        if not Path(original_path).exists():
            original_path = patch_path
        original_img = Image.open(original_path).convert("RGB")
        patch_tensor = self.patch_transform(patch_img) if self.patch_transform else self.to_tensor(patch_img)
        original_tensor = (
            self.original_transform(original_img)
            if self.original_transform
            else self.to_tensor(original_img)
        )
        return torch.cat([patch_tensor, original_tensor], dim=0), label


def build_resnet50_dual_input(num_classes: int = 4, pretrained: bool = True) -> nn.Module:
    """ResNet50 that accepts 6 channels: cropped-patch RGB plus original RGB."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    old_conv = model.conv1
    new_conv = nn.Conv2d(
        6,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=False,
    )
    with torch.no_grad():
        new_conv.weight[:, :3] = old_conv.weight
        new_conv.weight[:, 3:] = old_conv.weight
        new_conv.weight *= 0.5
    model.conv1 = new_conv
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
