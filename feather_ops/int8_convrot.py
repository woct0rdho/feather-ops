"""Int8 GEMM over a prepacked weight, for the two matrix shapes of a training step.

The weight layout is the one `library.convrot_int8` / ComfyUI ship as `int8_tensorwise` + `convrot`
(`weight` int8 `[N, K]` in the rotated basis, `weight_scale` f32 per output channel). Both GEMMs a step
needs read that single copy:

    forward   C[M, N] = A[M, K] @ W[N, K]^T    contracts the weight's input axis
    backward  C[M, K] = A[M, N] @ W[N, K]      contracts the weight's output axis

`prepack_int8_convrot_weight` turns the released `[N, K]` tensor into the 4x4 byte micro-tile layout the
kernels read. The transformation is a pure permutation of the int8 bytes (no requantization, `int8 x int8 ->
int32` is exact), so it is bit-identical and `unpack_int8_convrot_weight` restores the released form.
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


def prepack_int8_convrot_weight(weight: torch.Tensor) -> torch.Tensor:
    """`[N, K]` int8 ConvRot weight -> prepacked `[K // 4, N // 4, 4, 4]` micro-tiles.

    A pure permutation of the bytes: micro-tile `(k4, n4)` holds `W[n4 * 4 : n4 * 4 + 4, k4 * 4 : k4 * 4 + 4]`
    with `k` contiguous inside the tile, which is what makes one copy serve both GEMM directions.
    """
    if weight.dim() != 2:
        raise ValueError(f"weight must be 2D, got {tuple(weight.shape)}")
    n, k = weight.shape
    if n % 4 != 0 or k % 4 != 0:
        raise ValueError(f"both weight axes must be multiples of 4, got {tuple(weight.shape)}")
    if weight.dtype != torch.int8:
        raise ValueError(f"weight must be int8, got {weight.dtype}")
    return weight.view(n // 4, 4, k // 4, 4).permute(2, 0, 1, 3).contiguous()


def unpack_int8_convrot_weight(prepacked: torch.Tensor) -> torch.Tensor:
    """Inverse of `prepack_int8_convrot_weight`: back to the released `[N, K]` int8 form."""
    k4, n4, inner_n, inner_k = prepacked.shape
    if (inner_n, inner_k) != (4, 4):
        raise ValueError(f"prepacked weight must end in (4, 4), got {tuple(prepacked.shape)}")
    return prepacked.permute(1, 2, 0, 3).reshape(n4 * 4, k4 * 4)


def convrot_quantize_activation(
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
        weight_prepacked: `prepack_int8_convrot_weight` output (any shape, flat is fine).
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
