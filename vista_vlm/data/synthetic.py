import numpy as np
import cv2
import torch
import random
from typing import Tuple, Optional, Union
from PIL import Image

def generate_perlin_noise_mask(
    height: int,
    width: int,
    min_scale: int = 0,
    max_scale: int = 6,
    threshold: float = 0.5
) -> np.ndarray:
    """
    Generates multi-octave Perlin-style smooth continuous noise mask.
    Thresholds the noise to create realistic organic anomaly shapes.
    """
    noise = np.zeros((height, width), dtype=np.float32)
    scales = 2 ** np.arange(min_scale, max_scale)
    weights = 1.0 / scales

    for scale, weight in zip(scales, weights):
        grid_h = max(2, int(height / (scale * 8)))
        grid_w = max(2, int(width / (scale * 8)))
        rand_grid = np.random.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)
        octave = cv2.resize(rand_grid, (width, height), interpolation=cv2.INTER_CUBIC)
        noise += weight * octave

    # Normalize to [0, 1]
    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    
    # Threshold to binary mask
    binary_mask = (noise > threshold).astype(np.float32)
    
    # Morphological cleanup: remove tiny speckles
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    return binary_mask

def generate_scratch_and_crack_mask(height: int, width: int) -> np.ndarray:
    """
    Generates realistic hairline scratches and structural crack defects.
    Produces thin, branching, curvilinear trajectories with variable width.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    num_cracks = random.randint(1, 3)

    for _ in range(num_cracks):
        length = random.randint(30, max(40, min(height, width) // 3))
        sx = random.randint(width // 6, 5 * width // 6)
        sy = random.randint(height // 6, 5 * height // 6)
        angle = random.uniform(0, 2 * np.pi)

        points = [(sx, sy)]
        curr_x, curr_y = float(sx), float(sy)
        seg_len = 8.0

        for _ in range(int(length / seg_len)):
            angle += random.uniform(-0.4, 0.4)
            curr_x += seg_len * np.cos(angle)
            curr_y += seg_len * np.sin(angle)
            cx_clamped = int(np.clip(curr_x, 5, width - 5))
            cy_clamped = int(np.clip(curr_y, 5, height - 5))
            points.append((cx_clamped, cy_clamped))

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 3)
        cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=thickness)

    return (mask > 0.5).astype(np.float32)


def generate_contamination_mask(height: int, width: int) -> np.ndarray:
    """
    Generates clusters of subtle contamination spots, dust, and stains.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    cx = random.randint(width // 4, 3 * width // 4)
    cy = random.randint(height // 4, 3 * height // 4)
    num_spots = random.randint(3, 10)

    for _ in range(num_spots):
        sx = int(np.clip(cx + np.random.normal(0, 25), 5, width - 5))
        sy = int(np.clip(cy + np.random.normal(0, 25), 5, height - 5))
        r = random.randint(2, 8)
        cv2.circle(mask, (sx, sy), r, 1.0, -1)

    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return (mask > 0.3).astype(np.float32)


def generate_geometric_mask(height: int, width: int) -> np.ndarray:
    """
    Generates realistic geometric anomaly masks: scratches, dents, lines, or polygons.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    choice = random.choice(["scratch", "crack", "contamination", "dent", "polygon"])

    if choice == "crack":
        return generate_scratch_and_crack_mask(height, width)
    elif choice == "contamination":
        return generate_contamination_mask(height, width)
    elif choice == "scratch":
        # Draw curved polyline
        num_points = random.randint(3, 7)
        points = []
        cx, cy = random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4)
        for _ in range(num_points):
            cx += random.randint(-40, 40)
            cy += random.randint(-40, 40)
            cx = np.clip(cx, 10, width - 10)
            cy = np.clip(cy, 10, height - 10)
            points.append((cx, cy))
        
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        thickness = random.randint(2, 6)
        cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=thickness)

    elif choice == "dent":
        # Draw ellipse
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
        axes = (random.randint(10, 45), random.randint(10, 45))
        angle = random.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)

    else:
        # Draw small polygon
        num_vertices = random.randint(3, 6)
        cx, cy = random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4)
        pts = []
        for _ in range(num_vertices):
            x = cx + random.randint(-30, 30)
            y = cy + random.randint(-30, 30)
            pts.append([np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    # Slight blur to soften artificial hard edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return (mask > 0.3).astype(np.float32)


class SyntheticAnomalyGenerator:
    """
    Self-contained Synthetic Anomaly Generator (Phase 7).
    Generates synthetic industrial defects (scratches, stains, cracks, structural deformations)
    from normal training data with strictly matched ground-truth binary masks.
    """
    def __init__(
        self,
        opacity_range: Tuple[float, float] = (0.2, 0.95),
        anomaly_prob: float = 0.5,
        perlin_scale: int = 6,
        seed: Optional[int] = None
    ):
        self.opacity_range = opacity_range
        self.anomaly_prob = anomaly_prob
        self.perlin_scale = perlin_scale
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def __call__(
        self, 
        image: np.ndarray, 
        source_image: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Args:
            image: Normal RGB image as uint8 numpy array [H, W, 3]
            source_image: Optional second normal image for CutPaste texture perturbation
        Returns:
            perturbed_image: uint8 numpy array [H, W, 3]
            ground_truth_mask: float32 numpy array [H, W] with values in {0, 1}
            is_anomalous: bool
        """
        h, w = image.shape[:2]

        if random.random() > self.anomaly_prob:
            # Return original normal image with empty mask
            return image.copy(), np.zeros((h, w), dtype=np.float32), False

        # Generate anomaly mask (organic or geometric)
        if random.random() < 0.6:
            mask = generate_perlin_noise_mask(h, w, max_scale=self.perlin_scale, threshold=0.55)
        else:
            mask = generate_geometric_mask(h, w)

        # Fallback if mask is empty
        if mask.sum() < 20:
            mask = generate_geometric_mask(h, w)

        # Generate perturbation texture
        texture_mode = random.choice(["cutpaste", "noise", "color_jitter", "blur"])
        pert_img = image.copy().astype(np.float32)

        if texture_mode == "cutpaste" and source_image is not None:
            # CutPaste texture from source normal image
            src = cv2.resize(source_image, (w, h)).astype(np.float32)
            # Apply slight rotation
            m_rot = cv2.getRotationMatrix2D((w // 2, h // 2), random.randint(-45, 45), 1.0)
            src = cv2.warpAffine(src, m_rot, (w, h))
            pert_img = src

        elif texture_mode == "noise":
            # High-frequency noise texture
            noise = np.random.normal(0, 40, (h, w, 3)).astype(np.float32)
            pert_img = np.clip(pert_img + noise, 0, 255)

        elif texture_mode == "color_jitter":
            # Local color/brightness perturbation
            pert_img = pert_img * np.random.uniform(0.3, 1.8)
            pert_img = np.clip(pert_img, 0, 255)

        else:  # blur / out-of-focus defect
            pert_img = cv2.GaussianBlur(pert_img, (25, 25), 0)

        # Soft boundary blending to eliminate unnatural step-artifacts
        soft_mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
        mask_3d = np.expand_dims(soft_mask, axis=-1)  # [H, W, 1]
        alpha = random.uniform(*self.opacity_range)
        blended = (1.0 - alpha * mask_3d) * image.astype(np.float32) + (alpha * mask_3d) * pert_img
        anomalous_image = np.clip(blended, 0, 255).astype(np.uint8)

        return anomalous_image, mask.astype(np.float32), True
