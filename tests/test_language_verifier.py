import pytest
import numpy as np
from vista_vlm.models.language_verifier import LanguageVerifier

def test_language_verifier_initialization():
    verifier = LanguageVerifier(embed_dim=512)
    assert verifier.clip is not None

def test_roi_extraction():
    verifier = LanguageVerifier(roi_threshold=0.5, min_box_area=50)

    # Synthetic anomaly map with two distinct peaks
    amap = np.zeros((518, 518), dtype=np.float32)
    amap[50:100, 50:100] = 0.85 # Peak 1
    amap[300:360, 200:280] = 0.92 # Peak 2

    rois = verifier.extract_rois(amap, image_shape=(518, 518))
    assert len(rois) == 2

    # Verify bounding box format [x1, y1, x2, y2]
    for roi in rois:
        assert len(roi["box"]) == 4
        x1, y1, x2, y2 = roi["box"]
        assert x2 > x1
        assert y2 > y1
        assert roi["area"] >= 50
        assert roi["max_score"] > 0.5

def test_rank_candidates_and_ambiguity():
    verifier = LanguageVerifier()

    # Create dummy image and roi
    image_np = np.ones((518, 518, 3), dtype=np.uint8) * 150
    rois = [{
        "box": [50, 50, 150, 150],
        "area": 10000,
        "mean_score": 0.82,
        "max_score": 0.95
    }]
    category = "bottle"

    findings = verifier.verify_rois(
        image_np=image_np,
        rois=rois,
        category=category
    )

    assert len(findings) == 1
    finding = findings[0]
    assert "top_defect_label" in finding
    assert "top_similarity_score" in finding
    assert "confidence_margin" in finding
    assert "is_ambiguous" in finding
    assert len(finding["candidate_rankings"]) > 0

    # Cosine similarities must be in [-1, 1]
    for cand in finding["candidate_rankings"]:
        assert -1.0 <= cand["score"] <= 1.0
