"""ROI crop used by training patches and the local UI."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np
from PIL import Image


def crop_tumor_region(
    image: np.ndarray,
    mask: np.ndarray,
    margin: int = 10,
    output_size: Tuple[int, int] = (224, 224),
) -> Tuple[np.ndarray, np.ndarray]:
    """Crop the largest mask component plus margin, then resize.

    Returns ``(cropped_image, cropped_mask)`` at ``output_size``.
    """
    binary = (mask > 0).astype(np.uint8)
    num, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if num <= 1:
        img_resized = np.array(Image.fromarray(image).resize(output_size))
        return img_resized, np.zeros(output_size, dtype=np.uint8)

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _area = stats[largest]
    height, width = image.shape[:2]
    rmin = max(0, y - margin)
    cmin = max(0, x - margin)
    rmax = min(height, y + h + margin)
    cmax = min(width, x + w + margin)

    cropped_img = image[rmin:rmax, cmin:cmax]
    cropped_mask = binary[rmin:rmax, cmin:cmax] * 255
    cropped_img = np.array(Image.fromarray(cropped_img).resize(output_size))
    cropped_mask = np.array(
        Image.fromarray(cropped_mask).resize(output_size, resample=Image.NEAREST)
    )
    return cropped_img, cropped_mask
