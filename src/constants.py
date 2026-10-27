# Canonical dataset labels and folder-name aliases.

from typing import Dict, List

CLASS_NAMES = ["glioma", "meningioma", "pituitary", "notumor"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

FOLDER_ALIASES: Dict[str, List[str]] = {
    "glioma": ["glioma_tumor", "glioma", "Glioma", "Glioma_Tumor"],
    "meningioma": ["meningioma_tumor", "meningioma", "Meningioma", "Meningioma_Tumor"],
    "pituitary": [
        "pituitary_tumor",
        "pituitary",
        "Pituitary",
        "Pituitary_Tumor",
        "Pituitary tumor",
        "pituitary tumor",
    ],
    "notumor": ["no_tumor", "notumor", "Notumor", "No_Tumor", "no tumor"],
}

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif"}
