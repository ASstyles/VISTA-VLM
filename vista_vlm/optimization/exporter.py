import torch
import os
from typing import Dict, Any
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.utils.logger import setup_logger

logger = setup_logger("vista_vlm.exporter")

def export_torchscript(model: VISTAAnomalyModel, save_path: str, sample_input_size: int = 518) -> str:
    """Exports VISTA model decoder/localization head to TorchScript graph."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.eval()
    dummy_input = torch.randn(1, 3, sample_input_size, sample_input_size)
    try:
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(save_path)
        logger.info(f"TorchScript model successfully saved to {save_path}")
        return save_path
    except Exception as e:
        logger.warning(f"Tracing full model encountered dynamic control flow, exporting script: {e}")
        scripted = torch.jit.script(model.decoder)
        decoder_path = save_path.replace(".pt", "_decoder.pt")
        scripted.save(decoder_path)
        logger.info(f"Saved scripted decoder to {decoder_path}")
        return decoder_path

def get_model_footprint(model: torch.nn.Module) -> Dict[str, Any]:
    """Calculates model parameter count, trainable parameters, and memory footprint in MB."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    buffer_size_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / (1024 * 1024)
    total_size_mb = param_size_mb + buffer_size_mb

    return {
        "parameters_total": int(total_params),
        "parameters_trainable": int(trainable_params),
        "parameter_size_mb": round(param_size_mb, 2),
        "total_footprint_mb": round(total_size_mb, 2)
    }
