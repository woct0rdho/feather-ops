"""Int8 GEMM over a prepacked weight, for the two matrix shapes of a training step.

The weight layout is the one `library.convrot_int8` / ComfyUI ship as `int8_tensorwise` + `convrot`
(`weight` int8 `[N, K]` in the rotated basis, `weight_scale` f32 per output channel). Both GEMMs a step
needs read that single copy:

    forward   C[M, N] = A[M, K] @ W[N, K]^T    contracts the weight's input axis
    backward  C[M, K] = A[M, N] @ W[N, K]      contracts the weight's output axis

`prepack_int8_weight` turns the released `[N, K]` tensor into the 4x4 byte micro-tile layout the
kernels read. The transformation is a pure permutation of the int8 bytes (no requantization, `int8 x int8 ->
int32` is exact), so it is bit-identical and `unpack_int8_weight` restores the released form.
"""

import torch
from torch.library import custom_op

# Tile schedule per role, measured on the model cells of the int8 ConvRot plan (bench/benchmark_int8_convrot.py).
# Each is a row of `INT8_TILES`. The comment is the (block_m x block_n) box and thread count.
_TILE_TINY_M = (1, 2, 4, 1, 2)  # 16x64, 64 threads: M <= 16 (a single row, per-token passes)
_TILE_SMALL_M = (2, 2, 2, 2, 2)  # 64x64, 128 threads: 1-256 rows (prefix passes, short text rows)
_TILE_NARROW = (4, 2, 2, 4, 4)  # 256x128, 256 threads: narrow outputs, where a 256-wide tile is mostly tail
_TILE_WIDE = (4, 4, 2, 4, 4)  # 256x256, 512 threads: wide outputs, the bulk of the model flops
_TILE_MID = (2, 4, 2, 4, 2)  # 128x128, 256 threads: everything else
_TILE_TALL = (2, 4, 2, 4, 4)  # 128x256, 256 threads: narrow output over a long reduction


def select_int8_config(M: int, N: int, K: int, transposed_b: bool) -> tuple[int, int, int, int, int]:
    """Tile schedule for one (M, N, K) cell and direction.

    `N`/`K` are the weight's output/input axes. The forward produces `N` and the backward produces `K`, so the
    thresholds are on the output width of the call. M is the token count, which is what the image size moves,
    and the rules are ranges rather than fitted shapes, so an unseen M lands in the right class.
    """
    out = K if transposed_b else N
    contraction = N if transposed_b else K
    # Small M. A 128-row tile splits into a full launch plus a 64-row tail launch - two single-wave grids -
    # and at this size the per-CTA fill/drain is what dominates, so one 64x64 launch wins until the output is
    # wide enough for the 128x128 tile's larger N panel to amortize (the crossover sits at `out ~ 4096` for
    # M = 192-256 and `out ~ 8192` for M = 128. M <= 64 keeps 64x64 up to 24576, and M >= 384 is back on the
    # 128-row tile). Above the crossover the 256x256 tile loses, so the answer there is 128x128.
    if M <= 16:
        # A row or two: the block count is what matters, and a 16-row tile gives 4x the blocks of 64 rows.
        return _TILE_TINY_M
    if M <= 64:
        return _TILE_SMALL_M
    if M <= 128:
        return _TILE_SMALL_M if out <= 8192 else _TILE_MID
    if M <= 256:
        # `192 x 12288 x 4096` is 6% faster on 64x64, but `256 x 12288 x 4096` in the gradient direction is 30%
        # slower, so the crossover after which the 128x128 tile's N panel wins stays.
        return _TILE_SMALL_M if out <= 4096 else _TILE_MID
    if M <= 512:
        # 512 rows is four 128-row tiles, so the tile height is bought and the output width decides: the narrow
        # tile while the output is narrow (`512 x 2048 x 1024` is 33% faster on 64x64, `512 x 4096 x 4096` 4%),
        # the 128x128 tile once the output is 8192 wide or more, and the 256x256 tile when the reduction is long
        # or the output is enormous, where the weight traffic dominates again.
        if out >= 16384 or contraction >= 8192:
            return _TILE_WIDE
        if out >= 8192:
            return _TILE_MID
        return _TILE_SMALL_M
    if out <= 1280 and M >= 1024:
        # narrow output: a 256-wide tile is mostly tail, but it needs enough M tile rows to fill the CUs
        return _TILE_NARROW
    if out >= 6144:
        return _TILE_WIDE
    if out >= 3072 and contraction >= 2048:
        # mid-width output: the 256x256 tile halves the number of A panels, which is what the cells with a real
        # reduction axis want in both directions (Z-Image 3840x*, H3's out_proj/fc2 5376x*, MOSS 3072x*, and the
        # gradients of the same families).
        # Exception: a power-of-two contraction up to 8192 at a mid-size M aliases the L2 sets with a 256-wide
        # tile (Qwen-Image / Klein 4096x4096 at M=4112 prefers 128x128 while M=9232 prefers 256x256. A
        # 16384-wide contraction at the same M prefers 256x256, so the exception stops at 8192, and at
        # M = 512 the wide tile wins, so it starts at 1024).
        if 4096 <= contraction <= 8192 and (contraction & (contraction - 1)) == 0 and 1024 <= M < 8192:
            return _TILE_MID
        return _TILE_WIDE
    if out < 2048 and contraction >= 4096:
        # narrow output with a long reduction: the 128x256 tile keeps twice the A panel per block
        return _TILE_TALL
    return _TILE_MID


def prepack_int8_weight(weight: torch.Tensor) -> torch.Tensor:
    """`[N, K]` int8 ConvRot weight -> prepacked `[K // 4, N // 4, 4, 4]` micro-tiles.

    A pure permutation of the bytes: micro-tile `(k4, n4)` holds `W[n4 * 4 : n4 * 4 + 4, k4 * 4 : k4 * 4 + 4]`
    with `k` contiguous inside the tile, which is what makes one copy serve both GEMM directions.
    """
    n, k = weight.shape  # the view below is the layout check. The op validates dtype and extents
    return weight.view(n // 4, 4, k // 4, 4).permute(2, 0, 1, 3).contiguous()


def unpack_int8_weight(prepacked: torch.Tensor) -> torch.Tensor:
    """Inverse of `prepack_int8_weight`: back to the released `[N, K]` int8 form."""
    k4, n4, inner_n, inner_k = prepacked.shape
    if (inner_n, inner_k) != (4, 4):
        raise ValueError(f"prepacked weight must end in (4, 4), got {tuple(prepacked.shape)}")
    return prepacked.permute(1, 2, 0, 3).reshape(n4 * 4, k4 * 4)


def quantize_convrot_activation(
    x: torch.Tensor,
    rotate: bool = True,
    col_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ConvRot activation quantization: rotate by the group-256 Hadamard and quantize int8 per row.

    Args:
        x: `[M, K]` bf16/fp16 activations, `K` a multiple of 256. Not modified.
        rotate: apply the block-diagonal Hadamard (the forward path). `False` quantizes as-is, which is what the
            input gradient needs (`dx = (g @ W) @ H` is rotated after the GEMM).
        col_scale: optional `[K]` float32 multiplier applied before quantization, i.e. the weight's
            per-output-channel scale folded into the gradient quantizer.

    Returns:
        `(q int8 [M, K], scale float32 [M])`. `scale` is of the normalized rotated values, so a consumer
        dequantizes with `q * scale` exactly as for the weight path.

    This is the fused producer for the same recipe as `quantize_convrot_weight` (group Hadamard, per-row
    `absmax / 127`, int8 codes, one fp32 scale per row): the offline function rotates with a matmul and is
    what a checkpoint exporter uses, this one rotates in registers while quantizing and is what a step
    uses. The stored scale folds the rotation's 1/sqrt(group) normalization, which is why the two agree on
    the dequantized values to the last few codes of a bf16 rotation.
    """
    q = torch.empty(x.shape, device=x.device, dtype=torch.int8)
    scale = torch.empty(x.shape[0], device=x.device, dtype=torch.float32)
    torch.ops.feather_ops.convrot_quantize.default(
        x,
        None if col_scale is None else col_scale.reshape(-1).to(device=x.device, dtype=torch.float32),
        q,
        scale,
        rotate,
    )
    return q, scale


def _launch(
    a: torch.Tensor,
    weight_prepacked: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    transposed_b: bool,
    tile: tuple[int, int, int, int, int],
) -> None:
    torch.ops.feather_ops.int8_scaled_mm.default(
        a,
        weight_prepacked.reshape(-1),
        a_scale.reshape(-1).to(torch.float32),
        None if weight_scale is None else weight_scale.reshape(-1).to(torch.float32),
        bias,
        out,
        transposed_b,
        *tile,
    )


@custom_op("feather_ops::int8_scaled_mm_auto", mutates_args=("out",))
def _int8_scaled_mm_auto(
    a: torch.Tensor,
    weight_prepacked: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    transposed_b: bool,
) -> None:
    """The schedule heuristic behind an opaque boundary, so a compiled caller sees one node."""
    tile = select_int8_config(a.shape[0], weight_prepacked.numel() // a.shape[1], a.shape[1], transposed_b)
    _launch(a, weight_prepacked, a_scale, weight_scale, bias, out, transposed_b, tile)


def int8_scaled_mm(
    a: torch.Tensor,
    weight_prepacked: torch.Tensor,
    a_scale: torch.Tensor,
    weight_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    transposed_b: bool = False,
    tile: tuple[int, int, int, int, int] | None = None,
) -> torch.Tensor:
    """`A @ W^T` (default) or `A @ W` (`transposed_b=True`) with per-row and per-output-channel scales.

    Args:
        a: `[M, K]` int8, contraction axis contiguous.
        weight_prepacked: `prepack_int8_weight` output (any shape, flat is fine).
        a_scale: `[M]` float32 per-row activation scale.
        weight_scale: `[N_out]` float32 per-output-channel weight scale. Required for the forward. For
            `transposed_b=True` it must be `None` because the caller folds it into the activation
            quantization (a per-`n` factor cannot be pulled out of a sum over `n`).
        bias: `[N_out]` in `out_dtype`, forward only.
        out_dtype: `torch.bfloat16` or `torch.float16`.
        transposed_b: contract the weight's output axis instead of its input axis.
        tile: pin the schedule instead of using `select_int8_config`.

    Returns:
        `[M, N_out]` in `out_dtype`.
    """
    M, K = a.shape[0], a.shape[1]
    # `n_out` is derived from K here, so K must be usable as a divisor. Everything else the C++ op
    # validates (2-D, dtypes, contiguity, the weight's extent, the output layout).
    if K == 0:
        raise ValueError("a must have a nonzero contraction axis")
    n_out = weight_prepacked.numel() // K
    if transposed_b and weight_scale is not None:
        raise ValueError("transposed_b folds the weight scale into the activation quantization, so weight_scale must be None")

    out = torch.empty((M, n_out), device=a.device, dtype=out_dtype)
    if tile is None:
        _int8_scaled_mm_auto(a, weight_prepacked, a_scale, weight_scale, bias, out, transposed_b)
    else:
        _launch(a, weight_prepacked, a_scale, weight_scale, bias, out, transposed_b, tile)
    return out


CONVROT_GROUP = 256  # the kernels implement the regular Hadamard over 256 channels only

_HADAMARD_CACHE: dict[tuple[int, str, torch.dtype], torch.Tensor] = {}


def _is_power_of_four(size: int) -> bool:
    while size > 1 and size % 4 == 0:
        size //= 4
    return size == 1


def build_hadamard(size: int = CONVROT_GROUP, device="cpu", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The normalized regular Hadamard matrix of `size` (a power of four), cached.

    Built as the Kronecker power of `h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]` divided by
    `sqrt(size)`, so it is symmetric and orthogonal (`H @ H == I`) and the same matrix undoes a rotation.
    The entries are `+-1/sqrt(size)`, exactly representable for the group sizes in use, so any consumer
    that builds it this way lands on the same values.
    """
    if not _is_power_of_four(size):
        raise ValueError(f"the regular Hadamard size must be a power of four, got {size}")
    key = (size, str(device), dtype)
    cached = _HADAMARD_CACHE.get(key)
    if cached is not None:
        return cached
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=device)
    h, current = h4, 4
    while current < size:
        h = torch.kron(h, h4)
        current *= 4
    h = h / (size**0.5)
    _HADAMARD_CACHE[key] = h
    return h


def rotate(x: torch.Tensor, hadamard: torch.Tensor, groupsize: int = CONVROT_GROUP) -> torch.Tensor:
    """`(x grouped along the last axis) @ H^T`, the ConvRot rotation (and, H being orthogonal, its inverse)."""
    size = x.shape[-1]
    if size % groupsize:
        raise ValueError(f"last axis {size} is not divisible by the group size {groupsize}")
    grouped = x.reshape(*x.shape[:-1], size // groupsize, groupsize)
    out = torch.matmul(grouped, hadamard.T.to(dtype=x.dtype, device=x.device))
    return out.reshape(x.shape)


def quantize_rowwise_int8(x: torch.Tensor, scale: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row int8 quantization of an already-rotated tensor, with `scale = absmax / 127` by default.

    The codes are clamped to `[-127, 127]`, which is the range a scale derived from the row's absmax
    produces by itself (the absmax maps to +-127), so on that path the clamp is only a guard against
    rounding noise. It can bind only when the caller passes a stored scale, and there -127 is the right
    bound: it is the code of the absmax, it is what the kernels' producer uses, and it keeps the range
    symmetric. Allowing -128 would change the bytes for a stored scale without describing any value that
    `absmax / 127` can resolve.
    """
    if scale is None:
        scale = (x.abs().amax(dim=-1).float() / 127.0).clamp(min=1e-30)
    elif not isinstance(scale, torch.Tensor):
        scale = torch.tensor(scale, device=x.device, dtype=torch.float32)
    else:
        scale = scale.reshape(-1).to(device=x.device, dtype=torch.float32)
    q = torch.clamp(torch.round(x.float() / scale.reshape(-1, *([1] * (x.ndim - 1)))), -127.0, 127.0)
    return q.to(torch.int8), scale


def quantize_convrot_weight(
    weight: torch.Tensor,
    groupsize: int = CONVROT_GROUP,
    rotate_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Offline ConvRot weight quantization: `W_rot = W @ H^T` per group, then per-row int8.

    `rotate_dtype` is the dtype the rotation runs in. `None` (the default) uses the weight's dtype, which
    is what the comfy-quants exporter does. `torch.float32` is the stricter form used by sd-scripts,
    where rotating a bf16 weight in bf16 shifts a few codes.

    This is the offline counterpart of `quantize_convrot_activation`, which produces the same
    `(int8, per-row scale)` pair for an activation with the rotation fused into the quantizer.
    """
    hadamard = build_hadamard(groupsize, device=weight.device, dtype=torch.float32)
    rotated = rotate(weight.to(rotate_dtype or weight.dtype), hadamard, groupsize)
    return quantize_rowwise_int8(rotated)


def dequantize_convrot_weight(q: torch.Tensor, scale: torch.Tensor, groupsize: int = CONVROT_GROUP) -> torch.Tensor:
    """`int8 code * per-row scale`, rotated back to the original basis (`float32`)."""
    weight = q.float() * scale.reshape(-1, 1)
    hadamard = build_hadamard(groupsize, device=q.device, dtype=torch.float32)
    return rotate(weight, hadamard, groupsize)


def convrot_int8_supported(shape, groupsize: int = CONVROT_GROUP, backward: bool = True) -> tuple[bool, str]:
    """Whether the prepacked int8 kernels can run a `[N, K]` ConvRot weight, and why not if they cannot.

    `K` and `N` are the micro-tile axes (multiples of 4). The forward additionally needs `K` to be a
    multiple of the 16-wide MMA step, the input gradient needs `N` to be, because it contracts that axis.
    `backward=True` (the default) checks both so one weight serves both directions.
    """
    n, k = int(shape[0]), int(shape[1])
    if groupsize != CONVROT_GROUP:
        return False, f"only group size {CONVROT_GROUP} is implemented, got {groupsize}"
    if k % groupsize:
        return False, f"in_features {k} is not a multiple of the group size {groupsize}"
    if k % 16:
        return False, f"in_features {k} is not a multiple of the 16-wide reduction step"
    if n % 4:
        return False, f"out_features {n} is not a multiple of 4"
    if backward and n % 16:
        return False, f"out_features {n} is not a multiple of 16, which the input gradient contracts"
    return True, ""


def _weight_extent(weight: torch.Tensor, weight_scale: torch.Tensor) -> tuple[int, int]:
    """`(N, K)` of an int8 ConvRot weight, from the released or the prepacked form.

    Both layouts hold `N * K` bytes, and the scale has one entry per output channel, so the extent does
    not depend on which one the caller holds.
    """
    n_out = int(weight_scale.numel())
    if n_out == 0 or weight.numel() % n_out:
        raise ValueError(f"weight of {weight.numel()} bytes does not divide into {n_out} output channels")
    return n_out, int(weight.numel() // n_out)


def convrot_int8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.bfloat16,
    groupsize: int = CONVROT_GROUP,
    rotate_activation: bool = True,
    weight_packed: bool = False,
) -> torch.Tensor:
    """`x @ W^T + bias` on an int8 ConvRot weight: rotate and quantize `x`, int8 GEMM with both scales.

    The weight is the released `[N, K]` int8 tensor a checkpoint holds, or its prepacked form
    (`weight_packed=True`). This is forward only. The input gradient is `convrot_int8_dgrad`, because
    it contracts the other axis and folds the weight scale into the quantizer instead.
    """
    n_out, k_in = _weight_extent(weight, weight_scale)
    if x.shape[-1] != k_in:
        # not a kernel check: `n_out` is derived from the weight's extent, so a mismatch would silently
        # reinterpret the weight instead of failing
        raise ValueError(f"activation has {x.shape[-1]} features, the weight {k_in}")
    x_2d = x.reshape(-1, k_in)
    q, scale = quantize_convrot_activation(x_2d, rotate=rotate_activation)
    if not weight_packed:
        weight = prepack_int8_weight(weight)
    out = int8_scaled_mm(q, weight, scale, weight_scale.reshape(-1), bias, out_dtype, False)
    return out.reshape(*x.shape[:-1], n_out)


def convrot_int8_dgrad(
    grad_out: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype = torch.bfloat16,
    groupsize: int = CONVROT_GROUP,
    rotate_activation: bool = True,
    weight_packed: bool = False,
) -> torch.Tensor:
    """Input gradient on an int8 ConvRot weight: `g @ W`, then the rotation back to the input basis.

    The GEMM contracts the weight's output axis, so the per-output-channel scale cannot be applied after
    the sum. It is folded into the row-wise quantization of `grad_out` instead, which is the same identity
    torchao uses for `B.T` scaling. `rotate_activation` applies `x @ H^T` to the result, undoing the
    forward's rotation of the activation.
    """
    _, k_in = _weight_extent(weight, weight_scale)
    grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
    q, scale = quantize_convrot_activation(grad_2d, False, col_scale=weight_scale.reshape(-1))
    if not weight_packed:
        weight = prepack_int8_weight(weight)
    out = int8_scaled_mm(q, weight, scale, None, None, out_dtype, True)
    if rotate_activation:
        out = rotate(out, build_hadamard(groupsize, device=out.device, dtype=out.dtype), groupsize)
    return out.reshape(*grad_out.shape[:-1], k_in)
