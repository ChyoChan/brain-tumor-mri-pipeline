"""
Stage 0 – Data loading, splitting, pseudo-mask generation, segmentation dataset,
and post-segmentation ROI cropping.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.constants import CLASS_NAMES, CLASS_TO_IDX, FOLDER_ALIASES, IMAGE_EXTS
from src.utils import ensure_dir  # noqa: F401 – used downstream


# ------------------------------------------------------------------ #
#  Kaggle Brain Tumor MRI Dataset helpers
# ------------------------------------------------------------------ #

def discover_images(data_dir: str) -> List[Tuple[str, int]]:
    """Walk *data_dir* and return list of (image_path, class_idx).

    Handles multiple Kaggle folder-name conventions automatically
    (e.g. ``glioma/``, ``glioma_tumor/``, ``Glioma_Tumor/``).
    Searches both ``<data_dir>/Training/`` and
    ``<data_dir>/Classification/Training/`` layouts.
    """
    samples: List[Tuple[str, int]] = []
    seen: set = set()  # deduplicate on case-insensitive filesystems
    base = Path(data_dir)

    search_roots = [base]
    for sub in ("Classification", "classification"):
        if (base / sub).exists():
            search_roots.append(base / sub)

    for root in search_roots:
        for split_name in ("Training", "Testing"):
            split_dir = root / split_name
            if not split_dir.exists():
                continue
            for cls_name in CLASS_NAMES:
                cls_dir = None
                for alias in FOLDER_ALIASES[cls_name]:
                    candidate = split_dir / alias
                    if candidate.exists():
                        cls_dir = candidate
                        break
                if cls_dir is None:
                    continue
                for img_path in sorted(cls_dir.glob("*")):
                    if img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif"):
                        key = str(img_path.resolve()).lower()
                        if key not in seen:
                            seen.add(key)
                            samples.append((str(img_path), CLASS_TO_IDX[cls_name]))
    return samples


def stratified_split(
    samples: List[Tuple[str, int]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[list, list, list]:
    """80 / 10 / 10 stratified split."""
    paths, labels = zip(*samples)
    paths, labels = list(paths), list(labels)

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths, labels, test_size=1 - ratios[0], stratify=labels, random_state=seed
    )
    val_ratio = ratios[1] / (ratios[1] + ratios[2])
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=1 - val_ratio, stratify=temp_labels, random_state=seed
    )

    train = list(zip(train_paths, train_labels))
    val = list(zip(val_paths, val_labels))
    test = list(zip(test_paths, test_labels))
    return train, val, test


def stratified_split_pairs(
    pairs: List[Tuple[str, str]],
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[list, list, list]:
    """80/10/10 stratified split for (image_path, mask_path) pairs.

    Stratification is based on the parent directory name of each image,
    which corresponds to the tumor class in the Kaggle segmentation dataset.
    """
    strat_keys = [Path(img).parent.name for img, _ in pairs]

    train_pairs, temp_pairs, train_keys, temp_keys = train_test_split(
        pairs, strat_keys,
        test_size=1 - ratios[0],
        stratify=strat_keys,
        random_state=seed,
    )
    val_ratio = ratios[1] / (ratios[1] + ratios[2])
    val_pairs, test_pairs = train_test_split(
        temp_pairs,
        test_size=1 - val_ratio,
        stratify=temp_keys,
        random_state=seed,
    )
    return list(train_pairs), list(val_pairs), list(test_pairs)


# ------------------------------------------------------------------ #
#  Real mask discovery (paired image + mask files from dataset)
# ------------------------------------------------------------------ #


def discover_image_mask_pairs(
    data_dir: str,
    allow_pseudo_fallback: bool = False,
    pseudo_output_dir: Optional[str] = None,
) -> List[Tuple[str, str]]:
    """Discover real image-mask pairs from the segmentation dataset.

    Supports two common layouts:

    **Layout A – class subfolders with interleaved masks**::

        data_dir/
          Glioma/
            enh_1841.png          ← image
            enh_1841_mask.png     ← corresponding mask
          Meningioma/ ...
          Pituitary tumor/ ...

    **Layout B – separate images/ and masks/ directories**::

        data_dir/
          images/   (or images/train, images/val, …)
          masks/    (or masks/train, masks/val, …)

    Mask filename convention:  ``<stem>_mask.<ext>``  or same name in a
    parallel ``masks/`` tree.

    Parameters
    ----------
    data_dir : str
        Root of the segmentation dataset.  The function will also try
        ``<data_dir>/Segmentation`` and ``<data_dir>/segmentation`` if
        the given path does not directly contain recognisable content.

    If no real mask pairs are found and ``allow_pseudo_fallback`` is True,
    this function falls back to generating Otsu pseudo-masks from
    classification images discovered under ``data_dir``.

    Returns
    -------
    list of (image_path, mask_path)
        Every entry is a verified pair – both files exist on disk.
    """
    root = Path(data_dir)
    if not root.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # Try root first, then common nested folders (depth <= 2) to handle
    # datasets placed one level below "data/" or "dataset/".
    candidate_roots: List[Path] = [root]
    for sub in ("Segmentation", "segmentation"):
        subdir = root / sub
        if subdir.exists() and subdir.is_dir():
            candidate_roots.append(subdir)

    first_level = [d for d in root.iterdir() if d.is_dir()]
    candidate_roots.extend(first_level)
    for d in first_level:
        try:
            candidate_roots.extend(c for c in d.iterdir() if c.is_dir())
        except Exception:
            continue

    # De-duplicate while preserving order.
    seen = set()
    unique_roots: List[Path] = []
    for cand in candidate_roots:
        key = str(cand.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique_roots.append(cand)

    def _discover_pairs_in_root(search_root: Path) -> List[Tuple[str, str]]:
        pairs_local: List[Tuple[str, str]] = []

        # --------------------------------------------------------------
        # Layout A: class subfolders with *_mask.ext interleaved
        # --------------------------------------------------------------
        class_dirs = sorted(
            d for d in search_root.iterdir()
            if d.is_dir() and d.name.lower() not in ("images", "masks")
        )

        if class_dirs:
            for cls_dir in class_dirs:
                for img_path in sorted(cls_dir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    if "_mask" in img_path.stem:
                        continue

                    found = False
                    for ext in (img_path.suffix, ".png", ".jpg", ".jpeg"):
                        mask_candidate = cls_dir / f"{img_path.stem}_mask{ext}"
                        if mask_candidate.exists():
                            pairs_local.append((str(img_path), str(mask_candidate)))
                            found = True
                            break

                    if not found:
                        sibling = cls_dir.parent / "masks" / cls_dir.name / img_path.name
                        if sibling.exists():
                            pairs_local.append((str(img_path), str(sibling)))

            if pairs_local:
                return pairs_local

        # --------------------------------------------------------------
        # Layout B: images/ + masks/ (optionally split by train/val/test)
        # --------------------------------------------------------------
        img_root = search_root / "images"
        mask_root = search_root / "masks"

        if img_root.exists() and mask_root.exists():
            split_dirs = sorted(d for d in img_root.iterdir() if d.is_dir())
            search_dirs = split_dirs if split_dirs else [img_root]

            for sdir in search_dirs:
                mdir = mask_root / sdir.name if sdir != img_root else mask_root
                if not mdir.exists():
                    mdir = mask_root

                for img_path in sorted(sdir.iterdir()):
                    if img_path.suffix.lower() not in IMAGE_EXTS:
                        continue
                    candidates = [
                        mdir / img_path.name,
                        mdir / f"{img_path.stem}_mask{img_path.suffix}",
                        mdir / f"{img_path.stem}_mask.png",
                    ]
                    for cand in candidates:
                        if cand.exists():
                            pairs_local.append((str(img_path), str(cand)))
                            break

        return pairs_local

    # Try to discover real image-mask pairs in all candidate roots.
    for search_root in unique_roots:
        pairs = _discover_pairs_in_root(search_root)
        if pairs:
            return pairs

    # Fall back to pseudo masks from classification images.
    if allow_pseudo_fallback:
        for search_root in unique_roots:
            cls_samples = discover_images(str(search_root))
            if not cls_samples:
                continue

            pseudo_root = (
                Path(pseudo_output_dir)
                if pseudo_output_dir
                else (search_root / "_auto_pseudo_masks")
            )
            pairs = generate_pseudo_masks_for_split(cls_samples, str(pseudo_root))
            if pairs:
                print(
                    f"[discover_image_mask_pairs] No real masks found in {data_dir}; "
                    f"generated {len(pairs)} pseudo masks at {pseudo_root}."
                )
                return pairs

    raise FileNotFoundError(
        f"No image-mask pairs found in {data_dir}. "
        f"Expected either class subfolders with *_mask files, "
        f"or images/ + masks/ subdirectories."
    )


# ------------------------------------------------------------------ #
#  Pseudo-mask generation (Otsu fallback when real masks unavailable)
# ------------------------------------------------------------------ #

def generate_pseudo_mask(image_path: str, output_path: str) -> str:
    """Generate a binary pseudo-mask using Otsu thresholding + morphology.

    This is a fallback for when the SciDB pixel-level masks are not
    available.  The mask is *approximate* — good enough to bootstrap
    U-Net training but should be replaced with real annotations.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read {image_path}")

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, mask)
    return output_path


def generate_pseudo_masks_for_split(
    samples: List[Tuple[str, int]],
    output_dir: str,
) -> List[Tuple[str, str]]:
    """Generate Otsu pseudo-masks for all tumor-class images in *samples*.

    Returns list of (image_path, mask_path) pairs (``notumor`` is skipped).
    """
    pairs: List[Tuple[str, str]] = []
    output_dir = Path(output_dir)
    for img_path, label in samples:
        cls_name = CLASS_NAMES[label]
        if cls_name == "notumor":
            continue
        mask_name = Path(img_path).stem + "_mask.png"
        mask_path = output_dir / cls_name / mask_name
        generate_pseudo_mask(img_path, str(mask_path))
        pairs.append((img_path, str(mask_path)))
    return pairs


# ------------------------------------------------------------------ #
#  Albumentations transform builders
# ------------------------------------------------------------------ #

def get_seg_train_transforms(img_size: int = 256) -> A.Compose:
    """Strong augmentation pipeline for segmentation training."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1, scale_limit=0.15, rotate_limit=15,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5,
        ),
        A.ElasticTransform(
            alpha=120, sigma=120 * 0.05,
            border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.3,
        ),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


def get_seg_val_transforms(img_size: int = 256) -> A.Compose:
    """Minimal transform pipeline for validation / test."""
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.5], std=[0.5]),
        ToTensorV2(),
    ])


# ------------------------------------------------------------------ #
#  Segmentation Dataset for U-Net training
# ------------------------------------------------------------------ #

class SegmentationDataset(Dataset):
    """PyTorch dataset that yields (image_tensor, mask_tensor) pairs.

    Uses Albumentations for augmentation so that identical spatial
    transforms are applied to both image and mask.

    Parameters
    ----------
    pairs : list of (image_path, mask_path)
    img_size : spatial resolution to resize to (square)
    augment : if True, use strong training augmentations; otherwise
              only Resize + Normalize + ToTensorV2
    transform : explicit Albumentations Compose; overrides ``augment``
    """

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        img_size: int = 256,
        augment: bool = False,
        transform: Optional[A.Compose] = None,
    ):
        self.pairs = pairs
        self.img_size = img_size

        if transform is not None:
            self.transform = transform
        elif augment:
            self.transform = get_seg_train_transforms(img_size)
        else:
            self.transform = get_seg_val_transforms(img_size)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        if mask is None:
            raise FileNotFoundError(f"Cannot read mask: {mask_path}")

        mask = (mask > 127).astype(np.uint8) * 255

        augmented = self.transform(image=img, mask=mask)
        img_tensor = augmented["image"]            # (1, H, W) float32
        mask_tensor = augmented["mask"]             # (H, W) uint8

        mask_tensor = (mask_tensor > 127).float().unsqueeze(0)  # (1, H, W)

        return img_tensor, mask_tensor


# ------------------------------------------------------------------ #
#  Post-segmentation: crop tumor ROI from prediction mask
# ------------------------------------------------------------------ #

def crop_tumor_region(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 10,
    output_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the bounding box of the tumor region + margin, then resize.

    Returns (cropped_image, cropped_mask) resized to *output_size*.
    """
    from skimage.measure import label, regionprops
    from PIL import Image as PILImage

    labeled = label(mask)
    regions = regionprops(labeled)

    if len(regions) == 0:
        img_resized = np.array(PILImage.fromarray(image).resize(output_size))
        mask_resized = np.zeros(output_size, dtype=np.uint8)
        return img_resized, mask_resized

    largest = max(regions, key=lambda r: r.area)
    rmin, cmin, rmax, cmax = largest.bbox
    h, w = image.shape[:2]
    rmin = max(0, rmin - margin)
    cmin = max(0, cmin - margin)
    rmax = min(h, rmax + margin)
    cmax = min(w, cmax + margin)

    cropped_img = image[rmin:rmax, cmin:cmax]
    cropped_mask = mask[rmin:rmax, cmin:cmax]

    cropped_img = np.array(PILImage.fromarray(cropped_img).resize(output_size))
    cropped_mask = np.array(
        PILImage.fromarray(cropped_mask).resize(output_size, resample=PILImage.NEAREST)
    )
    return cropped_img, cropped_mask

# ------------------------------------------------------------------ #
#  Classification patches from predicted tumor ROIs
# ------------------------------------------------------------------ #

def infer_patch_folder_name(model) -> str:
    """Choose a processed-patch folder name from the segmentation model type."""
    model_name = type(model).__name__.lower()
    if "resunet" in model_name:
        return "patches_resunet"
    if "attention" in model_name:
        return "patches_attention_unet"
    if "unet" in model_name:
        return "patches_unet"
    return f"patches_{model_name}"


def generate_classification_patches(
    samples: List[Tuple[str, int]],
    seg_model,
    device,
    output_dir: str,
    output_size: Tuple[int, int] = (224, 224),
    margin: int = 20,
) -> List[Tuple[str, int]]:
    """Crop tumor patches for classification using a segmentation model.

    ``notumor`` images are resized without a mask. Empty predictions fall back
    to a full-image resize so the classifier still gets an example.
    """
    from PIL import Image as PILImage
    from tqdm.auto import tqdm

    from src.segmentation import predict_mask

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    patched: List[Tuple[str, int]] = []
    for img_path, label in tqdm(samples, desc="Creating patches", leave=False):
        img = np.array(PILImage.open(img_path).convert("L"))
        cls_name = CLASS_NAMES[label]

        if cls_name != "notumor" and seg_model is not None:
            mask = predict_mask(seg_model, img, device)
            if mask.sum() > 0:
                img_crop, _ = crop_tumor_region(
                    img, mask, margin=margin, output_size=output_size
                )
            else:
                img_crop = np.array(PILImage.fromarray(img).resize(output_size))
        else:
            img_crop = np.array(PILImage.fromarray(img).resize(output_size))

        save_dir = output_path / cls_name
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / Path(img_path).name
        PILImage.fromarray(img_crop).save(str(save_path))
        patched.append((str(save_path), label))
    return patched
