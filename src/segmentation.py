"""
Stage 1 – Standard 2D U-Net segmentation: model, loss, training, inference, metrics.

Architecture: 4-level encoder-decoder with skip connections and double-conv blocks.
Binary segmentation (tumor = 1, background = 0).
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from medpy.metric.binary import dc as dice_coefficient
from medpy.metric.binary import hd95 as hausdorff_distance_95

from src.utils import AverageMeter, EarlyStopping, fmt_time


# ------------------------------------------------------------------ #
#  U-Net building blocks
# ------------------------------------------------------------------ #

class DoubleConv(nn.Module):
    """(Conv2d → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool_conv(x)


class Up(nn.Module):
    """Upsample → concat skip → DoubleConv"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Pad if spatial sizes differ (non-power-of-2 inputs)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# ------------------------------------------------------------------ #
#  U-Net model (4 down / 4 up levels)
# ------------------------------------------------------------------ #

class UNet(nn.Module):
    """Standard 2D U-Net with 4 encoder/decoder levels.

    Parameters
    ----------
    in_channels  : number of input channels (1 for grayscale MRI)
    out_channels : number of output classes (1 for binary segmentation)
    base_filters : number of filters in the first level (doubled each level)
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.inc = DoubleConv(in_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16)
        self.up1 = Up(f * 16, f * 8)
        self.up2 = Up(f * 8, f * 4)
        self.up3 = Up(f * 4, f * 2)
        self.up4 = Up(f * 2, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


# ------------------------------------------------------------------ #
#  ResUNet model (residual encoder-decoder)
# ------------------------------------------------------------------ #

class ResidualBlock(nn.Module):
    """Two-conv residual block with optional projection shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = F.relu(out + identity, inplace=True)
        return out


class ResDown(nn.Module):
    """Downsampling residual block (stride-2)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ResidualBlock(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResUp(nn.Module):
    """Transpose-conv upsample, skip concat, then residual block."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.block = ResidualBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


class ResUNet(nn.Module):
    """Residual U-Net variant used in comparison notebooks."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, f, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(f),
            nn.ReLU(inplace=True),
        )
        self.enc1 = ResidualBlock(f, f)
        self.enc2 = ResDown(f, f * 2)
        self.enc3 = ResDown(f * 2, f * 4)
        self.enc4 = ResDown(f * 4, f * 8)
        self.enc5 = ResDown(f * 8, f * 16)
        self.dec4 = ResUp(f * 16, f * 8, f * 8)
        self.dec3 = ResUp(f * 8, f * 4, f * 4)
        self.dec2 = ResUp(f * 4, f * 2, f * 2)
        self.dec1 = ResUp(f * 2, f, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.stem(x)
        x2 = self.enc1(x1)
        x3 = self.enc2(x2)
        x4 = self.enc3(x3)
        x5 = self.enc4(x4)
        x6 = self.enc5(x5)
        x = self.dec4(x6, x5)
        x = self.dec3(x, x4)
        x = self.dec2(x, x3)
        x = self.dec1(x, x2)
        return self.outc(x)



# ------------------------------------------------------------------ #
#  Attention U-Net
# ------------------------------------------------------------------ #

class ConvBlock(nn.Module):
    """Two 3x3 convs with batch-norm and ReLU."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Additive attention gate for skip connections."""

    def __init__(self, g_ch: int, x_ch: int, inter_ch: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(g_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(x_ch, inter_ch, kernel_size=1, bias=True),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class AttentionUpBlock(nn.Module):
    """Upsample, attend to skip, then conv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(g_ch=out_ch, x_ch=skip_ch, inter_ch=max(out_ch // 2, 1))
        self.conv = ConvBlock(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        dy = skip.size(2) - x.size(2)
        dx = skip.size(3) - x.size(3)
        x = F.pad(x, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        skip = self.att(x, skip)
        return self.conv(torch.cat([skip, x], dim=1))


class AttentionUNet(nn.Module):
    """Attention U-Net used in the three-model segmentation comparison."""

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters
        self.enc1 = ConvBlock(in_channels, f)
        self.enc2 = ConvBlock(f, f * 2)
        self.enc3 = ConvBlock(f * 2, f * 4)
        self.enc4 = ConvBlock(f * 4, f * 8)
        self.bottleneck = ConvBlock(f * 8, f * 16)
        self.pool = nn.MaxPool2d(2)
        self.dec4 = AttentionUpBlock(f * 16, f * 8, f * 8)
        self.dec3 = AttentionUpBlock(f * 8, f * 4, f * 4)
        self.dec2 = AttentionUpBlock(f * 4, f * 2, f * 2)
        self.dec1 = AttentionUpBlock(f * 2, f, f)
        self.outc = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))
        x5 = self.bottleneck(self.pool(x4))
        x = self.dec4(x5, x4)
        x = self.dec3(x, x3)
        x = self.dec2(x, x2)
        x = self.dec1(x, x1)
        return self.outc(x)


def build_segmentation_model(
    name: str,
    in_channels: int = 1,
    out_channels: int = 1,
    base_filters: int = 64,
) -> nn.Module:
    """Factory for UNet, ResUNet, and AttentionUNet."""
    key = name.lower().replace("-", "").replace("_", "")
    if key == "unet":
        return UNet(in_channels, out_channels, base_filters)
    if key == "resunet":
        return ResUNet(in_channels, out_channels, base_filters)
    if key in {"attentionunet", "attunet"}:
        return AttentionUNet(in_channels, out_channels, base_filters)
    raise ValueError(f"Unknown segmentation model: {name}")

# ------------------------------------------------------------------ #
#  Loss function: Dice + BCE combo
# ------------------------------------------------------------------ #

class DiceBCELoss(nn.Module):
    """Combined Dice loss + Binary Cross-Entropy for binary segmentation.

    The two terms are weighted equally by default; adjust ``bce_weight``
    to shift the balance (e.g. 0.3 = mostly Dice-driven).
    """

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        pflat = probs.view(-1)
        tflat = targets.view(-1)
        intersection = (pflat * tflat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (
            pflat.sum() + tflat.sum() + self.smooth
        )

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


# ------------------------------------------------------------------ #
#  Training loop
# ------------------------------------------------------------------ #

def train_unet(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 100,
    lr: float = 1e-3,
    patience: int = 15,
    save_path: str = "models/segmentation/best_unet.pth",
) -> Dict[str, list]:
    """Full U-Net training loop with validation, early stopping, and model saving.

    Returns a history dict with per-epoch train_loss, val_loss, val_dice,
    val_iou, val_hd95, and HD95 invalid-rate diagnostics.
    """
    model = model.to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6,
    )
    early_stop = EarlyStopping(patience=patience)

    history: Dict[str, list] = {
        "train_loss": [],
        "val_loss": [],
        "val_dice": [],
        "val_iou": [],
        "val_hd95": [],
        "val_hd95_invalid_count": [],
        "val_hd95_invalid_rate": [],
        "lr": [],
    }
    best_dice = 0.0
    train_start = time.time()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 70}")
    print(f"  Training U-Net  |  {epochs} epochs  |  patience={patience}  |  device={device}")
    print(f"  Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")
    print(f"{'=' * 70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ---- Train ----
        model.train()
        loss_meter = AverageMeter()
        pbar = tqdm(train_loader, desc=f"  Train {epoch}/{epochs}", leave=False)
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            loss_meter.update(loss.item(), images.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        train_loss = loss_meter.avg

        # ---- Validate ----
        val_loss, val_metrics = _validate_unet(model, val_loader, criterion, device)
        val_dice = val_metrics["dice"]
        val_iou = val_metrics["iou"]
        val_hd95 = val_metrics["hd95"]
        val_hd95_invalid_count = int(val_metrics.get("hd95_invalid_count", 0))
        val_hd95_invalid_rate = float(val_metrics.get("hd95_invalid_rate", 0.0))

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        epoch_time = time.time() - epoch_start
        eta = epoch_time * (epochs - epoch)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_dice"].append(val_dice)
        history["val_iou"].append(val_iou)
        history["val_hd95"].append(val_hd95)
        history["val_hd95_invalid_count"].append(val_hd95_invalid_count)
        history["val_hd95_invalid_rate"].append(val_hd95_invalid_rate)
        history["lr"].append(current_lr)

        improved = val_dice > best_dice
        marker = " *" if improved else ""
        hd95_suffix = f" ({val_hd95_invalid_count} invalid)" if val_hd95_invalid_count else ""

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"loss: {train_loss:.4f}/{val_loss:.4f} | "
            f"dice={val_dice:.4f} | "
            f"iou={val_iou:.4f} | "
            f"hd95={val_hd95:.4f}{hd95_suffix} | "
            f"lr={current_lr:.2e} | "
            f"{fmt_time(epoch_time)}/ep  ETA {fmt_time(eta)}{marker}"
        )

        if improved:
            best_dice = val_dice
            torch.save(model.state_dict(), save_path)
            print(f"  -> Best model saved (Dice={best_dice:.4f})")

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    total_time = time.time() - train_start
    print(f"{'=' * 70}")
    print(f"  Training finished in {fmt_time(total_time)}  |  Best Dice={best_dice:.4f}")
    print(f"{'=' * 70}")

    return history


@torch.no_grad()
def _validate_unet(
    model: nn.Module,
    loader: DataLoader,
    criterion: DiceBCELoss,
    device: torch.device,
) -> Tuple[float, Dict[str, float]]:
    """Return (val_loss, mean_metrics) over the validation set."""
    model.eval()
    loss_meter = AverageMeter()
    dice_meter = AverageMeter()
    iou_meter = AverageMeter()
    hd95_values: List[float] = []

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        loss_meter.update(loss.item(), images.size(0))

        preds = (torch.sigmoid(logits) > 0.5).float()
        for i in range(preds.size(0)):
            p = preds[i].cpu().numpy().squeeze().astype(np.uint8)
            g = masks[i].cpu().numpy().squeeze().astype(np.uint8)
            sample_metrics = compute_segmentation_metrics(p, g)
            dice_meter.update(sample_metrics["dice"])
            iou_meter.update(sample_metrics["iou"])
            hd95_values.append(
                float(sample_metrics.get("hd95", sample_metrics.get("hausdorff95", float("inf"))))
            )

    hd95_array = np.asarray(hd95_values, dtype=np.float64)
    finite_hd95 = hd95_array[np.isfinite(hd95_array)]
    hd95_invalid_count = int((~np.isfinite(hd95_array)).sum())
    hd95_mean = float(np.mean(finite_hd95)) if finite_hd95.size else float("inf")

    metrics = {
        "dice": float(dice_meter.avg),
        "iou": float(iou_meter.avg),
        "hd95": hd95_mean,
        "hd95_invalid_count": hd95_invalid_count,
        "hd95_invalid_rate": float(hd95_invalid_count / hd95_array.size) if hd95_array.size else 0.0,
    }
    return loss_meter.avg, metrics


# ------------------------------------------------------------------ #
#  Inference
# ------------------------------------------------------------------ #

@torch.no_grad()
def predict_mask(
    model: nn.Module,
    image: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
) -> np.ndarray:
    """Run U-Net inference on a single grayscale image (H, W) → binary mask (H, W).

    The image is resized to 256×256 for the network, then the mask is
    resized back to the original spatial dimensions.

    Robustness: if the fixed threshold produces an empty mask, we apply an
    adaptive fallback threshold based on the confidence peak to reduce false
    empty predictions.
    """
    model.eval()
    h, w = image.shape[:2]

    img = image.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    img_tensor = F.interpolate(img_tensor, size=(256, 256), mode="bilinear", align_corners=False)
    img_tensor = img_tensor.to(device)

    logits = model(img_tensor)
    prob_256 = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)

    from PIL import Image as PILImage

    # Resize probability map back first (bilinear), then threshold at full resolution.
    prob_full = np.array(
        PILImage.fromarray(prob_256, mode="F").resize((w, h), resample=PILImage.BILINEAR),
        dtype=np.float32,
    )
    mask_full = (prob_full > threshold).astype(np.uint8)

    # Fallback for occasional empty predictions:
    # 1) use a softer threshold tied to the peak probability.
    if mask_full.sum() == 0 and float(prob_full.max()) > 0.0:
        adaptive_thr = float(np.clip(prob_full.max() * 0.5, 0.10, max(0.10, threshold - 1e-3)))
        mask_full = (prob_full > adaptive_thr).astype(np.uint8)

    # 2) if still empty, use top-probability region at full resolution.
    if mask_full.sum() == 0:
        q = float(np.quantile(prob_full, 0.995))
        mask_full = (prob_full >= q).astype(np.uint8)

    # Keep only largest connected component to reduce speckle noise.
    if mask_full.sum() > 0:
        try:
            from scipy import ndimage
            labeled, ncomp = ndimage.label(mask_full)
            if ncomp > 1:
                sizes = ndimage.sum(mask_full, labeled, index=np.arange(1, ncomp + 1))
                largest = int(np.argmax(sizes) + 1)
                mask_full = (labeled == largest).astype(np.uint8)
        except Exception:
            pass

    return mask_full


@torch.no_grad()
def predict_batch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> List[np.ndarray]:
    """Run inference on a DataLoader and return a list of binary masks."""
    model.eval()
    masks = []
    for images, _ in tqdm(loader, desc="Predicting", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).cpu().numpy()
        for i in range(preds.shape[0]):
            masks.append(preds[i].squeeze().astype(np.uint8))
    return masks


# ------------------------------------------------------------------ #
#  Segmentation evaluation metrics
# ------------------------------------------------------------------ #

def compute_segmentation_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    voxel_spacing: Tuple[float, ...] = (1.0, 1.0),
) -> Dict[str, float]:
    """Compute Dice, IoU, HD95, sensitivity, specificity for binary masks."""
    pred_bin = (pred > 0).astype(np.uint8)
    gt_bin = (gt > 0).astype(np.uint8)
    pred_fg = pred_bin.astype(bool)
    gt_fg = gt_bin.astype(bool)

    if gt_bin.sum() == 0 and pred_bin.sum() == 0:
        return {
            "dice": 1.0,
            "iou": 1.0,
            "hausdorff95": 0.0,
            "hd95": 0.0,
            "sensitivity": 1.0,
            "specificity": 1.0,
        }

    if gt_bin.sum() == 0 or pred_bin.sum() == 0:
        return {
            "dice": 0.0,
            "iou": 0.0,
            "hausdorff95": np.inf,
            "hd95": np.inf,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }

    dsc = dice_coefficient(pred_bin, gt_bin)
    intersection = np.logical_and(pred_fg, gt_fg).sum()
    union = np.logical_or(pred_fg, gt_fg).sum()
    iou = intersection / (union + 1e-8)

    try:
        hd = hausdorff_distance_95(pred_bin, gt_bin, voxelspacing=voxel_spacing)
    except Exception:
        hd = float("inf")

    tp = np.logical_and(pred_fg, gt_fg).sum()
    fn = np.logical_and(~pred_fg, gt_fg).sum()
    fp = np.logical_and(pred_fg, ~gt_fg).sum()
    tn = np.logical_and(~pred_fg, ~gt_fg).sum()

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)

    return {
        "dice": float(dsc),
        "iou": float(iou),
        "hausdorff95": float(hd),
        "hd95": float(hd),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
    }


def evaluate_segmentation(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Evaluate a U-Net model on a DataLoader. Returns (avg_metrics, std_metrics)."""
    model.eval()
    all_metrics: List[Dict[str, float]] = []

    for images, masks in tqdm(loader, desc="Evaluating segmentation", leave=False):
        images = images.to(device)
        with torch.no_grad():
            logits = model(images)
        preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
        gt = masks.cpu().numpy()

        for i in range(preds.shape[0]):
            m = compute_segmentation_metrics(preds[i].squeeze(), gt[i].squeeze())
            all_metrics.append(m)

    if not all_metrics:
        empty = {
            "dice": 0.0,
            "iou": 0.0,
            "hausdorff95": 0.0,
            "hd95": 0.0,
            "hd95_invalid_count": 0.0,
            "hd95_invalid_rate": 0.0,
            "sensitivity": 0.0,
            "specificity": 0.0,
        }
        return empty, empty

    avg = {
        k: float(np.mean([m[k] for m in all_metrics]))
        for k in all_metrics[0]
        if k not in {"hd95", "hausdorff95"}
    }
    std = {
        k: float(np.std([m[k] for m in all_metrics]))
        for k in all_metrics[0]
        if k not in {"hd95", "hausdorff95"}
    }

    hd95_values = np.asarray(
        [m.get("hd95", m.get("hausdorff95", float("inf"))) for m in all_metrics],
        dtype=np.float64,
    )
    finite_hd95 = hd95_values[np.isfinite(hd95_values)]
    hd95_invalid_count = int((~np.isfinite(hd95_values)).sum())

    hd95_avg = float(np.mean(finite_hd95)) if finite_hd95.size else float("inf")
    hd95_std = float(np.std(finite_hd95)) if finite_hd95.size else float("inf")

    avg["hd95"] = hd95_avg
    avg["hausdorff95"] = hd95_avg
    avg["hd95_invalid_count"] = float(hd95_invalid_count)
    avg["hd95_invalid_rate"] = float(hd95_invalid_count / hd95_values.size) if hd95_values.size else 0.0

    std["hd95"] = hd95_std
    std["hausdorff95"] = hd95_std
    std["hd95_invalid_count"] = 0.0
    std["hd95_invalid_rate"] = 0.0

    return avg, std
