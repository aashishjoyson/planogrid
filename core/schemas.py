"""Pydantic v2 contracts shared by every layer — models/, training/, inference/,
app/backend/, app/frontend/ all construct or consume these, never bare dicts or
tuples across a module boundary. See DESIGN.md §4.2.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, computed_field


class BBox(BaseModel):
    """Axis-aligned bounding box. Absolute pixel coordinates unless is_normalized."""

    x1: float
    y1: float
    x2: float
    y2: float
    is_normalized: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def area(self) -> float:
        return self.width * self.height

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @computed_field  # type: ignore[prop-decorator]
    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def to_absolute(self, image_width: int, image_height: int) -> BBox:
        """Return an absolute-pixel copy. No-op if already absolute."""
        if not self.is_normalized:
            return self
        return BBox(
            x1=self.x1 * image_width,
            y1=self.y1 * image_height,
            x2=self.x2 * image_width,
            y2=self.y2 * image_height,
            is_normalized=False,
        )


class DetectionClass(StrEnum):
    """The closed set of classes Planogrid's three specialist detectors emit.
    Deliberately closed (unlike OCR engine names) — see DESIGN.md ADR-001.
    """

    PRODUCT = "product"
    EMPTY_GAP = "empty_gap"
    PRICE_TAG = "price_tag"


class Detection(BaseModel):
    """One detector output: a class, a box, a confidence, and provenance."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    bbox: BBox
    class_name: DetectionClass
    confidence: float = Field(ge=0.0, le=1.0)
    detector_source: str = Field(
        description="Registry name of the detector that produced this, e.g. 'yolo26-product'"
    )


class OCRResult(BaseModel):
    """Raw + parsed OCR output for one price-tag crop."""

    raw_text: str
    parsed_price: float | None = None
    currency: str | None = None
    char_confidence: float = Field(ge=0.0, le=1.0)
    engine: str = Field(description="Registry name of the OCR engine that produced this")


class ShelfRow(BaseModel):
    """A horizontal band of detections clustered by y-centroid proximity."""

    row_index: int
    y_band: tuple[float, float]
    member_ids: list[str] = Field(
        default_factory=list, description="Detection.id values in this row"
    )


class Association(BaseModel):
    """A product detection matched to its price tag, if any was found."""

    product_id: str
    tag_id: str | None = None
    ocr_result: OCRResult | None = None
    match_cost: float | None = Field(
        default=None,
        description="Cost from the row-local Hungarian assignment; None if no candidate existed",
    )


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class ComplianceFlag(BaseModel):
    rule_name: str
    severity: Severity
    target_id: str = Field(
        description="The Detection.id or Association.product_id this flag is about"
    )
    message: str


class AnalysisResult(BaseModel):
    """The one object the pipeline produces, the API returns, and the UI renders.
    Never re-derived or duplicated into a parallel API-only or UI-only schema.
    """

    image_path: str
    image_width: int
    image_height: int
    analyzed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    detections: list[Detection] = Field(default_factory=list)
    rows: list[ShelfRow] = Field(default_factory=list)
    associations: list[Association] = Field(default_factory=list)
    flags: list[ComplianceFlag] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for d in self.detections:
            counts[d.class_name.value] = counts.get(d.class_name.value, 0) + 1
        counts["flags"] = len(self.flags)
        return counts
