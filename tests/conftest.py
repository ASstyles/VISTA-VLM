"""
Pytest Configuration and Reusable Test Fixtures for VISTA-VLM.
==============================================================
Provides:
  - Global pipeline singleton isolation and cleanup across tests.
  - Deterministic Mock Pipeline fixtures for API liveness and readiness testing.
  - Configurable Mock Pipeline Builder for granular error scenario testing.
"""

from __future__ import annotations

import time
from typing import Dict, Any, Optional
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
import torch.nn as nn

import vista_vlm.api.routes as routes
from vista_vlm.api.schemas import (
    InspectionResponse,
    InspectionStatus,
    ReliabilityIndicators,
)
from vista_vlm.models.vista_model import ModelState


class MockNominalMemory:
    """Mock nominal memory bank for testing readiness states without heavy tensor allocations."""
    def __init__(self, is_fitted: bool = True, size: int = 100, dim: int = 384):
        self._is_fitted = is_fitted
        self.patches = torch.randn(size, dim) if is_fitted else None
        self.ref_stats = {
            "median": 1.25,
            "mad": 0.35,
            "p95": 1.85,
            "p99": 2.50
        } if is_fitted else {}
        self.category = "bottle"
        self.k_neighbors = 3
        self.neighbor_aggregation = "mean"

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted and self.patches is not None and self.patches.numel() > 0


class MockConformalCalibrator:
    """Mock conformal calibrator with category threshold dictionary."""
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        self.category_thresholds = thresholds if thresholds is not None else {"bottle": 0.7755}
        self.alpha = 0.05
        self.sample_counts = {"bottle": 20}

    def is_calibrated(self, category: str = "default") -> bool:
        return category in self.category_thresholds

    def predict(self, score: float, category: str = "bottle", strict: bool = True):
        thresh = self.category_thresholds.get(category, 0.7755)
        is_anom = score >= thresh
        p_val = 0.01 if is_anom else 0.42
        return is_anom, p_val, thresh


class MockOODDetector:
    """Mock OOD detector with prototypes dictionary."""
    def __init__(self, is_fitted: bool = True):
        self.metric = "mahalanobis"
        self.threshold_percentile = 98.0
        self.prototypes = {
            "bottle": {
                "mean": np.zeros(384, dtype=np.float32),
                "cov_inv": np.eye(384, dtype=np.float32),
                "threshold": 3.75
            }
        } if is_fitted else {}

    def score(self, features, category: str = "bottle") -> float:
        return 1.25

    def is_ood(self, features, category: str = "bottle"):
        return False, 1.25, 3.75


class MockVISTAModel(nn.Module):
    """Mock neural visual anomaly model for API testing."""
    def __init__(
        self,
        is_ready: bool = True,
        state: ModelState = ModelState.READY,
        memory_fitted: bool = True,
        category: str = "bottle",
        has_memory_bank: bool = True
    ):
        super().__init__()
        self._is_ready = is_ready
        self.state = state
        self.category = category
        self.embed_dim = 384
        self.image_size = 518
        self.scoring_mode = "hybrid"
        self.memory_fusion_weight = 0.65
        self.backbone_name = "dinov2_vits14"
        self.top_k_ratio = 0.01
        self.global_head_weight = 0.35

        # Dummy parameter to satisfy get_model_footprint
        self.dummy_param = nn.Parameter(torch.zeros(10, 10))

        if has_memory_bank:
            self.nominal_memory = MockNominalMemory(is_fitted=memory_fitted)
        else:
            self.nominal_memory = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready and (self.state == ModelState.READY)

    @property
    def memory_bank(self) -> Optional[torch.Tensor]:
        if self.nominal_memory is not None:
            return self.nominal_memory.patches
        return None


class MockPipelineConfig:
    """Mock configuration tree."""
    def __init__(self):
        self.model = MagicMock()
        self.model.backbone = "dinov2_vits14"
        self.model.embed_dim = 384
        self.data = MagicMock()
        self.data.patch_size = 14
        self.data.grid_size = 37
        self.data.image_size = 518
        self.calibration = MagicMock()
        self.calibration.method = "split_conformal"
        self.calibration.target_false_alarm_rate = 0.05
        self.system = MagicMock()
        self.system.device = "cpu"


_UNSET = object()


class MockVISTAInspectionPipeline:
    """Mock end-to-end inspection pipeline fulfilling the production API contract."""
    def __init__(
        self,
        model: Any = _UNSET,
        calibrator: Any = _UNSET,
        ood_detector: Any = _UNSET,
        device: str = "cpu"
    ):
        self.model = MockVISTAModel() if model is _UNSET else model
        self.calibrator = MockConformalCalibrator() if calibrator is _UNSET else calibrator
        self.ood_detector = MockOODDetector() if ood_detector is _UNSET else ood_detector
        self.device = torch.device(device)
        self.config = MockPipelineConfig()

    def inspect(self, image_input, category: str = "bottle") -> InspectionResponse:
        return InspectionResponse(
            inspection_id=f"test_insp_{int(time.time())}",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            category=category,
            status=InspectionStatus.NORMAL,
            image_anomaly_score=0.15,
            is_anomaly=False,
            threshold=0.7755,
            num_defects_found=0,
            bounding_boxes=[],
            region_findings=[],
            textual_interpretation="Defect-free baseline container surface verified.",
            reliability=ReliabilityIndicators(
                is_calibrated=True,
                calibrated_threshold=0.7755,
                conformal_p_value=0.45,
                target_false_alarm_rate=0.05,
                is_ood=False,
                ood_distance=1.25,
                ood_threshold=3.75,
                in_uncertainty_band=False,
                operational_state="NORMAL",
                disposition="ACCEPT",
                human_review_recommended=False,
                review_reason="Nominal normal surface under conformal bounds"
            ),
            execution_time_ms=45.2
        )


@pytest.fixture(autouse=True)
def isolate_pipeline_singleton():
    """Restores routes.pipeline_instance after each test to avoid cross-test contamination."""
    orig = routes.pipeline_instance
    yield
    routes.pipeline_instance = orig


@pytest.fixture
def mock_ready_pipeline(monkeypatch):
    """Provides a fully initialized, valid READY mock pipeline."""
    pipe = MockVISTAInspectionPipeline()
    monkeypatch.setattr(routes, "get_pipeline", lambda: pipe)
    return pipe


@pytest.fixture
def mock_pipeline_builder(monkeypatch):
    """Factory fixture for constructing pipelines with selective flaws."""
    def _builder(
        has_model: bool = True,
        model_is_ready: bool = True,
        model_state: ModelState = ModelState.READY,
        has_memory_bank: bool = True,
        memory_fitted: bool = True,
        has_calibrator: bool = True,
        calibrator_thresholds: Optional[Dict[str, float]] = None,
        has_ood_detector: bool = True,
        ood_fitted: bool = True,
    ) -> MockVISTAInspectionPipeline:
        model = None
        if has_model:
            model = MockVISTAModel(
                is_ready=model_is_ready,
                state=model_state,
                memory_fitted=memory_fitted,
                has_memory_bank=has_memory_bank
            )

        calibrator = None
        if has_calibrator:
            thresholds = {"bottle": 0.7755} if calibrator_thresholds is None else calibrator_thresholds
            calibrator = MockConformalCalibrator(thresholds=thresholds)

        ood = None
        if has_ood_detector:
            ood = MockOODDetector(is_fitted=ood_fitted)

        pipe = MockVISTAInspectionPipeline(
            model=model,
            calibrator=calibrator,
            ood_detector=ood
        )
        monkeypatch.setattr(routes, "get_pipeline", lambda: pipe)
        return pipe

    return _builder
