import os
import tempfile
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib_vista"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve

def plot_roc_pr_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str,
    title_prefix: str = "VISTA-VLM"
):
    """Generates dual-panel ROC and Precision-Recall plots."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2563eb", lw=2, label="ROC Curve")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", linestyle="--")
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel("False Positive Rate (FPR)")
    axes[0].set_ylabel("True Positive Rate (TPR)")
    axes[0].set_title(f"{title_prefix} — Receiver Operating Characteristic")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right")

    # PR Curve
    axes[1].plot(recall, precision, color="#10b981", lw=2, label="Precision-Recall Curve")
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title(f"{title_prefix} — Precision-Recall Curve")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

def plot_calibration_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: str,
    num_bins: int = 10
):
    """Generates reliability diagram showing confidence vs accuracy."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_centers = 0.5 * (bin_boundaries[:-1] + bin_boundaries[1:])
    accuracies = []
    confidences = []

    for i in range(num_bins):
        in_bin = (y_probs >= bin_boundaries[i]) & (y_probs < bin_boundaries[i + 1])
        if np.sum(in_bin) > 0:
            accuracies.append(np.mean(y_true[in_bin]))
            confidences.append(np.mean(y_probs[in_bin]))
        else:
            accuracies.append(bin_centers[i])
            confidences.append(bin_centers[i])

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#94a3b8", label="Perfect Calibration")
    plt.bar(bin_centers, accuracies, width=0.08, alpha=0.6, color="#6366f1", edgecolor="#4338ca", label="Empirical Accuracy")
    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Anomaly Rate")
    plt.title("Reliability Diagram (Calibration)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
