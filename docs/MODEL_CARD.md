# VISTA-VLM Model Card

## Model Details
- **Model Name**: VISTA-VLM (Vision Intelligence through Spatial and Textual Alignment)
- **Version**: 1.0.0
- **Model Type**: Multimodal Foundation Anomaly Localization & Verification Model
- **Primary Visual Backbone**: Frozen DINOv2 ViT-S/14 ($518 \times 518$ input resolution, $37 \times 37$ grid, 384 embedding dims, blocks 4, 8, 12)
- **Language Verifier**: Frozen CLIP ViT-B/16 (512-dim embedding, prompt ranking)
- **Trainable Components**: Global-Local FiLM modulation, progressive conv decoder, hybrid anomaly head (1,950,418 trainable parameters)
- **License**: MIT License

---

## Intended Use
- **Primary Use**: Automated surface inspection, defect localization, and semantic defect classification for industrial manufacturing parts (bottles, cables, capsules, tiles, PCBs).
- **Intended Users**: Quality assurance engineers, manufacturing plant technicians, computer vision researchers.
- **Operating Modes**:
  - Autonomous Pass/Fail (committed only outside the conformal uncertainty band)
  - Human-in-the-Loop Escalation (for OOD, uncertain, or ambiguous defect cases)

---

## Out-of-Scope Use
- Not intended for medical diagnostic imaging without specialized clinical re-calibration.
- Not intended for autonomous safety-critical aerospace flight-hardware clearance without redundant multi-sensor certification.
- Never use under arbitrary uncalibrated lighting or focal length changes without running conformal re-calibration.

---

## Performance Metrics (MVTec AD Bottle)
- **Image AUROC**: 1.0000
- **Image Average Precision (AP)**: 1.0000
- **Pixel AUROC**: 0.5625
- **Pixel AUPRO**: 0.6385
- **Empirical False Alarm Rate**: 0.0000 (Target $\alpha = 0.05$)
- **Expected Calibration Error**: 0.0497
- **Inference Latency (CPU)**: 600.8 ms P50 latency

---

## Ethical Considerations & Reliability Guarantees
- **Conformal Validity**: Guarantees finite-sample false-alarm bounds $\le \alpha$ under exchangeability.
- **Explainability**: Localizes defect bounding boxes and reports confidence margins between competing textual defect descriptions.
- **Feedback Protection**: Human review submissions are logged in independent audit records and do not mutate production model weights without rigorous regression validation.
