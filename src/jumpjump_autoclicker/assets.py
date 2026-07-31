from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(slots=True)
class TemplateImage:
    image: np.ndarray
    mask: np.ndarray | None = None
    path: Path | None = None


def load_template(path: Path) -> TemplateImage | None:
    """Load a template image and extract alpha as an OpenCV mask when present."""
    if not path.exists():
        return None

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None

    if len(image.shape) == 3 and image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]
        mask = (alpha > 0).astype(np.uint8) * 255
        return TemplateImage(image=bgr, mask=mask, path=path)

    return TemplateImage(image=image, path=path)


def load_templates(directory: Path) -> list[TemplateImage]:
    """Load all PNG templates from a directory in stable filename order."""
    if not directory.exists():
        return []

    templates: list[TemplateImage] = []
    for path in sorted(directory.glob("*.png")):
        template = load_template(path)
        if template is not None:
            templates.append(template)
    return templates

