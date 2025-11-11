import torch 

def select_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_broadcast(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Make a (C,) tensor broadcastable to ref shape (e.g. (B,C,T,H,W) or (C,T,H,W))."""
    # target broadcast shape (1, C, 1, 1) or (1, C, 1, 1, 1) depending on ref dims
    if t.ndim != 1:
        raise ValueError("expect per-channel 1D tensor")
    # create (1, C, 1, 1)
    b = t.view(1, -1, 1, 1)
    if ref.ndim == 5:  # (B,C,T,H,W)
        # expand to (1,C,1,1,1) then rely on broadcasting across time
        b = b.unsqueeze(2)  # (1,C,1,1,1)
    return b.to(device=ref.device, dtype=ref.dtype)