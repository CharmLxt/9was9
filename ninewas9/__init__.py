"""9was9 — универсальный ускоритель numpy-вычислений."""
from .version import __version__
from .kvcache import KVCache
from .ropecache import RoPECache
from .fused_linear import (
    FusedLinear, FusedMLP, fuse_qkv, fuse_gate_up,
)
from .fast_softmax import (
    fast_softmax, log_softmax,
    apply_causal_mask, scaled_dot_product_attention,
    top_k_filter, top_p_filter,
)
from . import image_ops
from .image_ops import (
    fast_resize, block_downsample, rgb_to_grayscale,
    rgb_to_bgr, normalize_image, pad_image,
)

__all__ = [
    "__version__",
    "KVCache",
    "RoPECache",
    "FusedLinear", "FusedMLP", "fuse_qkv", "fuse_gate_up",
    "fast_softmax", "log_softmax",
    "apply_causal_mask", "scaled_dot_product_attention",
    "top_k_filter", "top_p_filter",
    "image_ops",
    "fast_resize", "block_downsample", "rgb_to_grayscale",
    "rgb_to_bgr", "normalize_image", "pad_image",
]