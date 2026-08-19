"""One DatasetAdapter subclass per source. See base.py for the shared interface."""

from datasets.adapters.base import DatasetAdapter, DatasetSpec, FetchResult
from datasets.adapters.direct import DirectAdapter
from datasets.adapters.huggingface import HuggingFaceAdapter
from datasets.adapters.kaggle import KaggleAdapter
from datasets.adapters.roboflow import RoboflowAdapter

ADAPTERS: dict[str, type[DatasetAdapter]] = {
    "kaggle": KaggleAdapter,
    "roboflow": RoboflowAdapter,
    "huggingface": HuggingFaceAdapter,
    "direct": DirectAdapter,
}

__all__ = ["ADAPTERS", "DatasetAdapter", "DatasetSpec", "FetchResult"]
