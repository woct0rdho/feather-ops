import torch


def _is_hipblaslt_n_or_t_layout(x: torch.Tensor) -> bool:
    rows, cols = x.shape
    return (x.stride(0) == 1 and x.stride(1) == rows) or (x.stride(1) == 1 and x.stride(0) == cols)


def _validate_inputs(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype) -> None:
    if out_dtype != torch.float16:
        raise RuntimeError("hipBLASLt fp16 wrapper only supports out_dtype=torch.float16")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError("a and b must be CUDA tensors")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise RuntimeError("a and b must be float16")
    if a.dim() != 2 or b.dim() != 2:
        raise RuntimeError("a and b must be 2D")
    if a.shape[1] != b.shape[0]:
        raise RuntimeError(f"incompatible matmul shapes: {tuple(a.shape)} x {tuple(b.shape)}")
    if not _is_hipblaslt_n_or_t_layout(a):
        raise RuntimeError(f"a must be contiguous in hipBLASLt N or T layout, got strides {a.stride()}")
    if not _is_hipblaslt_n_or_t_layout(b):
        raise RuntimeError(f"b must be contiguous in hipBLASLt N or T layout, got strides {b.stride()}")


@torch.library.custom_op("feather_ops_internal::mm_hipblaslt_fp16", mutates_args=())
def _op(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    out = torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.mm_hipblaslt_fp16.default(
        a,
        b,
        scale,
        bias,
        out,
        alpha_scalar,
        use_relu,
        solution_index,
    )
    return out


@_op.register_fake
def _(
    a: torch.Tensor,
    b: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    alpha_scalar: float,
    use_relu: bool,
    solution_index: int,
) -> torch.Tensor:
    return torch.empty_strided((a.shape[0], b.shape[1]), (1, a.shape[0]), device=a.device, dtype=out_dtype)


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
    _validate_inputs(a, b, out_dtype)
    alpha_scalar = 1.0
    if scale is not None and scale.numel() == 1:
        alpha_scalar = float(scale.item())
        scale = None

    return _op(
        a,
        b,
        scale,
        bias,
        out_dtype,
        alpha_scalar,
        use_relu,
        solution_index,
    )


def to_col_major(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_strided(x.shape, (1, x.shape[0]), device=x.device, dtype=x.dtype)
    out.copy_(x)
    return out
