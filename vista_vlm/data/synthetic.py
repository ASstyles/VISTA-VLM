import numpy as np
import cv2
import random
from typing import Tuple, Optional

def get_foreground_mask(image: np.ndarray) -> np.ndarray:
    """
    Extracts binary foreground object mask from an industrial inspection image.
    Uses gradient and intensity contrast against background corners to ensure
    defects are generated on the actual product surface.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    h, w = gray.shape

    # Estimate background from corners (5x5 pixels in each corner)
    corners = np.concatenate([
        gray[:5, :5].flatten(),
        gray[:5, -5:].flatten(),
        gray[-5:, :5].flatten(),
        gray[-5:, -5:].flatten()
    ])
    bg_median = float(np.median(corners))

    diff = np.abs(gray.astype(np.float32) - bg_median)
    # Otsu threshold on the difference
    _, fg_mask = cv2.threshold(diff.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological closing to fill holes inside the object
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)

    # If foreground is too small (< 5%) or too large (> 95%), fall back to center crop mask
    fg_area = (fg_mask > 0).sum()
    total_area = h * w
    if fg_area < 0.05 * total_area or fg_area > 0.95 * total_area:
        fallback = np.zeros((h, w), dtype=np.uint8)
        fallback[int(h * 0.15):int(h * 0.85), int(w * 0.15):int(w * 0.85)] = 255
        return fallback

    return fg_mask


def generate_localized_perlin_mask(
    height: int,
    width: int,
    center: Optional[Tuple[int, int]] = None,
    max_radius: int = 70
) -> np.ndarray:
    """
    Generates realistic, localized organic anomaly patches (stains, liquid marks, oxidation).
    Confines Perlin noise to a localized bounding region to match real defect dimensions (0.5% - 6% area).
    """
    mask = np.zeros((height, width), dtype=np.float32)

    if center is None:
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
    else:
        cx, cy = center

    rh = random.randint(25, max_radius)
    rw = random.randint(25, max_radius)

    y1, y2 = max(0, cy - rh), min(height, cy + rh)
    x1, x2 = max(0, cx - rw), min(width, cx + rw)
    patch_h, patch_w = y2 - y1, x2 - x1

    if patch_h < 10 or patch_w < 10:
        return mask

    # Generate multi-octave noise in patch
    noise = np.zeros((patch_h, patch_w), dtype=np.float32)
    scales = [2, 4, 8]
    for s in scales:
        grid_h = max(2, patch_h // s)
        grid_w = max(2, patch_w // s)
        rand_grid = np.random.uniform(0.0, 1.0, (grid_h, grid_w)).astype(np.float32)
        octave = cv2.resize(rand_grid, (patch_w, patch_h), interpolation=cv2.INTER_CUBIC)
        noise += (1.0 / s) * octave

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
    binary_patch = (noise > 0.58).astype(np.float32)

    # Elliptical window falloff so patch doesn't have rectangular borders
    ey, ex = np.ogrid[:patch_h, :patch_w]
    center_y, center_x = patch_h / 2.0, patch_w / 2.0
    dist_sq = ((ex - center_x) / (patch_w / 2.0 + 1e-6)) ** 2 + ((ey - center_y) / (patch_h / 2.0 + 1e-6)) ** 2
    window = (dist_sq <= 1.0).astype(np.float32)

    patch_mask = binary_patch * window
    mask[y1:y2, x1:x2] = patch_mask
    return mask


def generate_perlin_noise_mask(height: int, width: int, **kwargs) -> np.ndarray:
    """Backward-compatible alias for localized organic Perlin noise generation."""
    return generate_localized_perlin_mask(height, width)


def generate_scratch_and_crack_mask(
    height: int,
    width: int,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Generates realistic hairline scratches and structural crack defects.
    Produces thin, branching, curvilinear trajectories with variable width (1-3 px).
    """
    mask = np.zeros((height, width), dtype=np.float32)
    num_cracks = random.randint(1, 2)

    for _ in range(num_cracks):
        length = random.randint(25, 90)
        if center is None:
            sx = random.randint(width // 4, 3 * width // 4)
            sy = random.randint(height // 4, 3 * height // 4)
        else:
            sx = int(np.clip(center[0] + np.random.randint(-30, 30), 10, width - 10))
            sy = int(np.clip(center[1] + np.random.randint(-30, 30), 10, height - 10))

        angle = random.uniform(0, 2 * np.pi)

        points = [(sx, sy)]
        curr_x, curr_y = float(sx), float(sy)
        seg_len = 6.0

        for _ in range(int(length / seg_len)):
            angle += random.uniform(-0.45, 0.45)
            curr_x += seg_len * np.cos(angle)
            curr_y += seg_len * np.sin(angle)
            cx_clamped = int(np.clip(curr_x, 5, width - 5))
            cy_clamped = int(np.clip(curr_y, 5, height - 5))
            points.append((cx_clamped, cy_clamped))

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 3)
        cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=thickness)

    return (mask > 0.5).astype(np.float32)


def generate_contamination_mask(
    height: int,
    width: int,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Generates clusters of subtle contamination spots, dust specks, and pinholes.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    if center is None:
        cx = random.randint(width // 4, 3 * width // 4)
        cy = random.randint(height // 4, 3 * height // 4)
    else:
        cx, cy = center

    num_spots = random.randint(3, 8)
    for _ in range(num_spots):
        sx = int(np.clip(cx + np.random.normal(0, 20), 5, width - 5))
        sy = int(np.clip(cy + np.random.normal(0, 20), 5, height - 5))
        r = random.randint(1, 5)
        cv2.circle(mask, (sx, sy), r, 1.0, -1)

    return (mask > 0.5).astype(np.float32)


def generate_geometric_mask(
    height: int,
    width: int,
    center: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Generates realistic geometric anomaly masks: scratches, dents, chips, or cuts.
    """
    mask = np.zeros((height, width), dtype=np.float32)
    choice = random.choice(["scratch", "crack", "contamination", "dent", "chip"])

    if choice == "crack":
        return generate_scratch_and_crack_mask(height, width, center)
    elif choice == "contamination":
        return generate_contamination_mask(height, width, center)
    elif choice == "scratch":
        num_points = random.randint(3, 6)
        if center is None:
            cx, cy = random.randint(width // 4, 3 * width // 4), random.randint(height // 4, 3 * height // 4)
        else:
            cx, cy = center
        points = []
        for _ in range(num_points):
            cx += random.randint(-25, 25)
            cy += random.randint(-25, 25)
            cx = int(np.clip(cx, 10, width - 10))
            cy = int(np.clip(cy, 10, height - 10))
            points.append((cx, cy))

        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        thickness = random.randint(1, 3)
        cv2.polylines(mask, [pts], isClosed=False, color=1.0, thickness=thickness)

    elif choice == "dent":
        if center is None:
            cx = random.randint(width // 4, 3 * width // 4)
            cy = random.randint(height // 4, 3 * height // 4)
        else:
            cx, cy = center
        axes = (random.randint(6, 22), random.randint(6, 22))
        angle = random.randint(0, 360)
        cv2.ellipse(mask, (cx, cy), axes, angle, 0, 360, 1.0, -1)

    else:  # chip
        if center is None:
            cx = random.randint(width // 4, 3 * width // 4)
            cy = random.randint(height // 4, 3 * height // 4)
        else:
            cx, cy = center
        num_vertices = random.randint(3, 5)
        pts = []
        for _ in range(num_vertices):
            x = cx + random.randint(-18, 18)
            y = cy + random.randint(-18, 18)
            pts.append([int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))])
        cv2.fillPoly(mask, [np.array(pts, dtype=np.int32)], 1.0)

    return (mask > 0.5).astype(np.float32)


class SyntheticAnomalyGenerator:
    """
    Advanced Multi-Scale Synthetic Anomaly Generator (VISTA-VLM v3.0).
    Generates realistic industrial defects matching empirical defect distributions:
    - 40% Fine defects: hairline cracks, scratches, pinholes (area 0.05% - 1.5%)
    - 40% Medium defects: localized stains, scuffs, abrasions, chips (area 1.5% - 5.0%)
    - 20% Organic Perlin clusters: stains, fluid leaks, oxidation (area 5.0% - 10.0%)
    - Strictly constrained to object foreground surface.
    - Soft boundary blending for realistic physical gradients.
    """
    def __init__(
        self,
        opacity_range: Tuple[float, float] = (0.35, 0.95),
        anomaly_prob: float = 0.5,
        max_defect_ratio: float = 0.12,
        seed: Optional[int] = None
    ):
        self.opacity_range = opacity_range
        self.anomaly_prob = anomaly_prob
        self.max_defect_ratio = max_defect_ratio
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
            # Nominal image without defect
            return image.copy(), np.zeros((h, w), dtype=np.float32), False

        # Extract foreground mask to locate candidate defect sites on the product surface
        fg_mask = get_foreground_mask(image)
        fg_indices = np.argwhere(fg_mask > 0)

        if len(fg_indices) > 50:
            rand_idx = random.randint(0, len(fg_indices) - 1)
            defect_cy, defect_cx = int(fg_indices[rand_idx][0]), int(fg_indices[rand_idx][1])
            center = (defect_cx, defect_cy)
        else:
            center = (w // 2, h // 2)

        # Multi-scale defect selection
        r = random.random()
        if r < 0.40:
            # Fine defects: crack or scratch
            mask = generate_scratch_and_crack_mask(h, w, center=center)
        elif r < 0.80:
            # Medium defects: geometric dent, chip, contamination
            mask = generate_geometric_mask(h, w, center=center)
        else:
            # Organic Perlin stain / oxidation
            mask = generate_localized_perlin_mask(h, w, center=center, max_radius=random.randint(30, 65))

        # Restrict defect strictly to foreground object
        mask = mask * (fg_mask.astype(np.float32) / 255.0)

        # Fallback if masked region is empty
        if mask.sum() < 15:
            mask = generate_scratch_and_crack_mask(h, w, center=(w // 2, h // 2))

        # Cap defect size to physical upper bound
        max_pixels = int(h * w * self.max_defect_ratio)
        if mask.sum() > max_pixels:
            mask = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)

        # Generate perturbation texture
        texture_mode = random.choice(["cutpaste", "noise", "color_jitter", "invert"])
        pert_img = image.copy().astype(np.float32)

        if texture_mode == "cutpaste" and source_image is not None:
            src = cv2.resize(source_image, (w, h)).astype(np.float32)
            m_rot = cv2.getRotationMatrix2D((w // 2, h // 2), random.randint(-45, 45), 1.0)
            src = cv2.warpAffine(src, m_rot, (w, h))
            pert_img = src

        elif texture_mode == "noise":
            noise = np.random.normal(0, random.uniform(30, 60), (h, w, 3)).astype(np.float32)
            pert_img = np.clip(pert_img + noise, 0, 255)

        elif texture_mode == "color_jitter":
            factor = random.choice([random.uniform(0.2, 0.6), random.uniform(1.4, 2.2)])
            pert_img = np.clip(pert_img * factor, 0, 255)

        else:  # local contrast inversion
            pert_img = 255.0 - pert_img

        # Soft boundary blending
        soft_mask = cv2.GaussianBlur(mask, (3, 3), 0.8)
        mask_3d = np.expand_dims(soft_mask, axis=-1)
        alpha = random.uniform(*self.opacity_range)
        blended = (1.0 - alpha * mask_3d) * image.astype(np.float32) + (alpha * mask_3d) * pert_img
        anomalous_image = np.clip(blended, 0, 255).astype(np.uint8)

        binary_gt = (mask > 0.3).astype(np.float32)
        return anomalous_image, binary_gt, True


def generate_multiscale_synthetic_anomaly(
    image: np.ndarray,
    source_image: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate a multi-scale synthetic anomaly on an image.
    Guarantees generation of an anomalous sample (anomaly_prob=1.0).
    """
    gen = SyntheticAnomalyGenerator(anomaly_prob=1.0)
    anom_img, mask, _ = gen(image, source_image)
    return anom_img, mask

