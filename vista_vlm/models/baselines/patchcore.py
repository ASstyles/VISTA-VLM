import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional
from vista_vlm.models.backbones.dinov2 import dinov2_vits14
from vista_vlm.models.memory_bank import coreset_approx_greedy_k_center, coreset_greedy_k_center
from vista_vlm.utils.registry import BASELINE_REGISTRY

@BASELINE_REGISTRY.register("patchcore")
class PatchCore(nn.Module):
    """
    PatchCore Anomaly Detection Baseline (Phase 3 & V6.0).
    Constructs a genuine Greedy k-Center coreset memory bank of intermediate patch features.
    Computes k-NN distance maps at test time for pixel- and image-level anomaly scoring.
    """
    def __init__(
        self,
        embed_dim: int = 384,
        coreset_sampling_ratio: float = 0.10,
        image_size: int = 518,
        k_neighbors: int = 3,
        use_approx_k_center: bool = True,
        **kwargs
    ):
        super().__init__()
        self.image_size = image_size
        self.coreset_ratio = kwargs.get("coreset_ratio", coreset_sampling_ratio)
        self.k_neighbors = kwargs.get("n_neighbors", k_neighbors)
        self.use_approx_k_center = use_approx_k_center
        self.encoder = dinov2_vits14(img_size=image_size, out_indices=[4, 8])
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

        self.memory_bank: Optional[torch.Tensor] = None

    def fit(self, dataloader: torch.utils.data.DataLoader, device: str = "cpu"):
        """Extracts patch features from normal training images and builds coreset memory bank."""
        self.encoder.to(device)
        patch_list = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(device)
                feat_dict = self.encoder(images)
                # Concatenate blocks 4 and 8: [B, 2*embed_dim, 37, 37]
                feats = torch.cat(feat_dict["patch_features"], dim=1)
                B, C, H, W = feats.shape
                # Reshape to [B * H * W, C]
                flat_patches = feats.permute(0, 2, 3, 1).reshape(-1, C)
                patch_list.append(flat_patches.cpu())

        all_patches = torch.cat(patch_list, dim=0) # [N_patches, C]
        total_p = all_patches.shape[0]
        n_coreset = max(10, int(total_p * self.coreset_ratio))

        # Genuine Greedy k-Center coreset selection
        if self.use_approx_k_center:
            coreset_t, _ = coreset_approx_greedy_k_center(all_patches, n_coreset, proj_dim=64, seed=42)
        else:
            coreset_t, _ = coreset_greedy_k_center(all_patches, n_coreset, seed=42)

        self.memory_bank = coreset_t.to(device)

    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computes anomaly score and spatial anomaly map.
        Args:
            x: Input image tensor [B, 3, H, W]
        Returns:
            image_score: [B]
            anomaly_map: [B, 1, H, W]
        """
        if self.memory_bank is None:
            raise RuntimeError("PatchCore memory bank is empty. Call fit() first.")

        device = x.device
        self.encoder.to(device)
        mem = self.memory_bank.to(device)

        with torch.no_grad():
            feat_dict = self.encoder(x)
            feats = torch.cat(feat_dict["patch_features"], dim=1) # [B, C, 37, 37]
            B, C, H, W = feats.shape
            patches = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # Compute min distance from each patch to memory bank
            dist_maps = []
            for b in range(B):
                p_b = patches[b] # [H*W, C]
                # Efficient pairwise distance: ||a - b||^2 = a^2 + b^2 - 2ab
                dists = torch.cdist(p_b, mem, p=2.0) # [H*W, N_mem]
                min_dists, _ = torch.topk(dists, k=self.k_neighbors, dim=1, largest=False)
                score_per_patch = min_dists[:, 0] # Nearest neighbor distance
                dist_map = score_per_patch.reshape(1, 1, H, W)
                dist_maps.append(dist_map)

            dist_tensor = torch.cat(dist_maps, dim=0) # [B, 1, H, W]
            # Interpolate raw distance tensor to input resolution
            anomaly_map = F.interpolate(dist_tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)

            # Image score: maximum patch distance in un-normalized map (Roth et al., 2022)
            # Preserves strict cross-image distance ranking without destructive per-image z-score distortion
            image_score = anomaly_map.view(B, -1).max(dim=1)[0]

        return {
            "image_score": image_score,
            "anomaly_map": anomaly_map
        }
