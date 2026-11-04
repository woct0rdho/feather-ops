import torch


def mm_hipblaslt_fp16(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    *,
    use_relu: bool = False,
    solution_index: int = -1,
) -> torch.Tensor:
    """fp16 hipBLASLt matmul with an optional per-row scale, bias and ReLU epilogue.

    `a` and `b` must be contiguous in hipBLASLt's N or T layout and `out_dtype` is always
    `torch.float16`. The native op validates all of that, so nothing is repeated here. The result
    is column-major (hipBLASLt's C layout), i.e. `out.stride() == (1, M)`.

    The output buffer is allocated here, outside the `out=` custom op, so a compiled caller can
    pool and reuse it.
    """
    alpha_scalar = 1.0
    if scale is not None and scale.numel() == 1:
        # A one-element scale is the alpha scalar. The kernel expects a full M-element row scale.
        alpha_scalar = float(scale.item())
        scale = None

    out = torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.mm_hipblaslt_fp16.default(a, b, scale, bias, out, alpha_scalar, use_relu, solution_index)
    return out
