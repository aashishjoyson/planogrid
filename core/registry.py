"""Generic name -> class registry. Detectors and OCR engines register themselves
via decorator at import time and are selected by string from YAML — callers never
import a concrete implementation directly. Adding a new engine is one new file
plus one decorator; nothing here or in inference/ changes. See DESIGN.md §4.3, ADR-005.

core/ stays dependency-free of models/: this file is generic over `type[Any]`
rather than importing the Detector/OCREngine ABCs it registers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

from core.exceptions import ComponentAlreadyRegisteredError, ComponentNotRegisteredError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def _decorator(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ComponentAlreadyRegisteredError(
                    f"{self._kind} '{name}' is already registered to "
                    f"{self._items[name].__module__}.{self._items[name].__qualname__}"
                )
            self._items[name] = cls
            return cls

        return _decorator

    def get(self, name: str) -> type[T]:
        try:
            return self._items[name]
        except KeyError as exc:
            available = (
                ", ".join(sorted(self._items)) or "<none registered — is the module imported?>"
            )
            raise ComponentNotRegisteredError(
                f"Unknown {self._kind} '{name}'. Available: {available}"
            ) from exc

    def available(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


# Two module-level singletons — one per swappable-component family.
detector_registry: Registry[Any] = Registry("detector")
ocr_registry: Registry[Any] = Registry("OCR engine")


def register_detector(name: str) -> Callable[[type[T]], type[T]]:
    return detector_registry.register(name)


def register_ocr(name: str) -> Callable[[type[T]], type[T]]:
    return ocr_registry.register(name)
