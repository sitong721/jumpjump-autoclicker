from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


class DebugWriter:
    def __init__(self, enabled: bool, output_dir: Path) -> None:
        self.enabled = enabled
        self.output_dir = output_dir
        if self.enabled:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(self, prefix: str, image: np.ndarray) -> None:
        if not self.enabled:
            return

        timestamp = datetime.now().strftime("%H%M%S_%f")[:-3]
        cv2.imwrite(str(self.output_dir / f"{prefix}_{timestamp}.png"), image)

