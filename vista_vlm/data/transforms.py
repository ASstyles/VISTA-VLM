import torch
import torchvision.transforms.functional as TF
from PIL import Image
from typing import Tuple, Optional, Union
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class VISTATransform:
    """
    Standard preprocessing for VISTA-VLM.
    Resizes images to target resolution (e.g. 518x518), applies ImageNet normalization,
    and formats masks as single-channel binary tensors (0.0 or 1.0).
    """
    def __init__(self, image_size: int = 518, is_training: bool = False):
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else image_size
        self.is_training = is_training
        self.mean = IMAGENET_MEAN
        self.std = IMAGENET_STD

    def __call__(
        self,
        image: Union[Image.Image, np.ndarray, torch.Tensor],
        mask: Optional[Union[Image.Image, np.ndarray, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert image to PIL if numpy
        if isinstance(image, np.ndarray):
            if image.ndim == 2:
                image = Image.fromarray(image).convert("RGB")
            else:
                image = Image.fromarray(image.astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            if image.ndim == 3 and image.shape[0] in (1, 3):
                # CxHxW -> HxWxC numpy
                image = Image.fromarray((image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

        # Ensure RGB
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            orig_w, orig_h = image.size
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Resize image
        img_resized = TF.resize(image, self.image_size, interpolation=TF.InterpolationMode.BICUBIC, antialias=True)
        img_tensor = TF.to_tensor(img_resized)
        img_norm = TF.normalize(img_tensor, mean=self.mean, std=self.std)

        # Handle mask
        if mask is not None:
            if isinstance(mask, np.ndarray):
                # If mask is float in [0, 1] or uint8 in {0, 1}, scale to 255 so TF.to_tensor correctly maps back to [0, 1]
                m_np = mask.astype(np.float32)
                if m_np.max() <= 1.0 and m_np.max() > 0:
                    m_np = m_np * 255.0
                mask = Image.fromarray(m_np.astype(np.uint8))
            elif isinstance(mask, torch.Tensor):
                m_t = mask.squeeze().float()
                if m_t.max() <= 1.0 and m_t.max() > 0:
                    m_t = m_t * 255.0
                mask = Image.fromarray(m_t.cpu().numpy().astype(np.uint8))

            # Nearest neighbor for segmentation mask to preserve sharp defect boundaries
            mask_resized = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)
            mask_tensor = TF.to_tensor(mask_resized)
            # Ensure single channel binary mask
            if mask_tensor.shape[0] > 1:
                mask_tensor = mask_tensor[:1, :, :]
            mask_binary = (mask_tensor > 0.2).float()
        else:
            mask_binary = torch.zeros((1, self.image_size[0], self.image_size[1]), dtype=torch.float32)

        return img_norm, mask_binary

def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts normalized tensor back to displayable uint8 numpy image."""
    t = tensor.detach().cpu().clone()
    if t.ndim == 4:
        t = t.squeeze(0)
    for c in range(3):
        t[c] = t[c] * IMAGENET_STD[c] + IMAGENET_MEAN[c]
    t = torch.clamp(t, 0.0, 1.0)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
