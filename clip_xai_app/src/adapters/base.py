from __future__ import annotations

from abc import ABC, abstractmethod

from ..contracts import AdapterHealth, PredictionResult


class BaseAdapter(ABC):
    model_id: str

    @abstractmethod
    def health(self) -> AdapterHealth:
        raise NotImplementedError

    @abstractmethod
    def predict_by_scan_id(self, scan_id: str) -> PredictionResult:
        raise NotImplementedError
