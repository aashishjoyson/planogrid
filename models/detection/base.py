"""Detector ABC — every detector implementation (YOLO-backed or otherwise)
speaks this one interface: an image in, a list of core.schemas.Detection
out. See DESIGN.md §4.3, §7.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from core.schemas import Detection


class Detector(ABC):
    """`detector_source` identifies which registered detector produced a
    given Detection — set on every Detection this implementation returns.
    """

    detector_source: str

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """`image` is a BGR uint8 array (OpenCV convention), shape HxWx3."""
