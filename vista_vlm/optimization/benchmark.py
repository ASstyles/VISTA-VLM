import time
import torch
import numpy as np
from typing import Dict, Any, List
from vista_vlm.models.vista_model import VISTAAnomalyModel
from vista_vlm.optimization.exporter import get_model_footprint

def benchmark_inference(
    model: VISTAAnomalyModel,
    input_size: int = 518,
    batch_sizes: List[int] = [1, 2, 4],
    num_warmup: int = 3,
    num_runs: int = 10,
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Profiles inference latency (P50, P90, P95, mean, min, max) and throughput (FPS).
    """
    model.to(device)
    model.eval()

    results = {
        "footprint": get_model_footprint(model),
        "device": device,
        "input_resolution": f"{input_size}x{input_size}",
        "batch_profiles": {}
    }

    with torch.no_grad():
        for bs in batch_sizes:
            dummy_input = torch.randn(bs, 3, input_size, input_size, device=device)

            # Warmup
            for _ in range(num_warmup):
                _ = model(dummy_input)

            # Timed runs
            latencies = []
            for _ in range(num_runs):
                t0 = time.perf_counter()
                _ = model(dummy_input)
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000.0) # total batch time in ms

            latencies = np.array(latencies)
            per_sample_latencies = latencies / bs

            results["batch_profiles"][f"batch_{bs}"] = {
                "batch_size": bs,
                "batch_latency_mean_ms": round(float(np.mean(latencies)), 2),
                "per_sample_p50_ms": round(float(np.percentile(per_sample_latencies, 50)), 2),
                "per_sample_p95_ms": round(float(np.percentile(per_sample_latencies, 95)), 2),
                "throughput_fps": round(float(1000.0 / np.mean(per_sample_latencies)), 2)
            }

    return results
