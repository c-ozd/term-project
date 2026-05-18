import gc
import torch


def free_vram():
    """
    Release all PyTorch-held GPU memory back to the OS.
    gc.collect() clears Python-side references; empty_cache() forces
    PyTorch to release its CUDA memory pool back to the OS.
    """
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()


def gpu_available() -> bool:
    """Check if CUDA GPU is available for inference."""
    available = torch.cuda.is_available()
    if available:
        gpu_name = torch.cuda.get_device_name(0)
        total_mb = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"  GPU available: {gpu_name} ({total_mb:.0f} MB)")
    else:
        print("  No GPU available — all inference on CPU.")
    return available
