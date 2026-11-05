import torch  # noqa: F401 Import torch before loading the extension.

from . import _C  # noqa: F401 Register native operators before torch.ops use. # ty: ignore[unresolved-import]
from .conv2d import conv3x3_prepacked, prepack_conv2d_weight_3x3
from .conv2d_module import Conv2d
from .int8_convrot import (
    convrot_int8_dgrad,
    convrot_int8_linear,
    dequantize_convrot_weight,
    int8_scaled_mm,
    prepack_int8_weight,
    quantize_convrot_activation,
    quantize_convrot_weight,
    unpack_int8_weight,
)
from .scaled_mm import scaled_mm_hip

__all__ = [
    "Conv2d",
    "conv3x3_prepacked",
    "convrot_int8_dgrad",
    "convrot_int8_linear",
    "dequantize_convrot_weight",
    "int8_scaled_mm",
    "prepack_conv2d_weight_3x3",
    "prepack_int8_weight",
    "quantize_convrot_activation",
    "quantize_convrot_weight",
    "scaled_mm_hip",
    "unpack_int8_weight",
]
