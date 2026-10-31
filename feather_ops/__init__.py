import torch  # noqa: F401 Import torch before loading the extension.

from . import _C  # noqa: F401 Register native operators before torch.ops use. # ty: ignore[unresolved-import]
from .scaled_mm import scaled_mm_hip

__all__ = ["scaled_mm_hip"]
