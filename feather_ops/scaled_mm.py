"""Mixed-precision matmul against a prepacked fp8 weight: `Y = (X @ W^T) * scale + bias`.

`X` is fp16 or bf16, `W` is fp8e5m2 or fp8e4m3 stored as `[K/16, N, 16]` (see
`prepack_b_for_scaled_mm`), `scale` is a per-tensor fp32 scalar and `bias` is optional. The output is
fp16 or bf16. All shape handling (M tails, N and K remainders) is inside the kernel, so any `M >= 1`
works as long as `N` and `K` are multiples of 16.

The tile is chosen by a static heuristic keyed on `(M, N, K)` and the dtypes. There is no runtime
search. `SCALED_MM_TILES` mirrors the instantiation list in `csrc/hip_kernel.cu`.
"""

import torch

# (block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n)
SCALED_MM_TILES: list[tuple[int, int, int, int, int]] = [
    (1, 1, 2, 1, 2),
    (1, 1, 4, 1, 2),
    (1, 2, 2, 1, 2),
    (1, 2, 4, 1, 2),
    (1, 4, 2, 1, 2),
    (1, 4, 4, 1, 2),
    (1, 8, 2, 1, 2),
    (1, 8, 4, 1, 2),
    (1, 1, 2, 2, 2),
    (1, 1, 4, 2, 2),
    (1, 1, 2, 4, 4),
    (1, 1, 4, 4, 4),
    (1, 2, 2, 2, 2),
    (1, 2, 4, 2, 2),
    (2, 1, 2, 2, 2),
    (2, 1, 4, 2, 2),
    (2, 2, 2, 2, 2),
    (1, 4, 2, 4, 2),
    (1, 4, 4, 4, 2),
    (1, 8, 2, 8, 2),
    (1, 8, 4, 8, 2),
    (2, 2, 2, 4, 4),
    (2, 2, 4, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 4, 4, 2),
    (2, 4, 2, 4, 4),
    (2, 4, 4, 4, 4),
    (4, 2, 2, 2, 4),
    (4, 2, 4, 2, 4),
    (4, 4, 2, 4, 2),
    (4, 4, 2, 4, 4),
]

# Tile shapes the heuristic can return. Each is an alias for a (warps_m, warps_n, unroll_k, repeat_m,
# repeat_n) row of SCALED_MM_TILES, named by the (block_m x block_n) box it covers.
_SHAPE_16x256 = (1, 8, 4, 1, 2)
_SHAPE_32x32 = (1, 2, 2, 2, 2)
_SHAPE_32x128 = (1, 2, 4, 1, 2)
_SHAPE_64x64 = (2, 2, 2, 2, 2)
_SHAPE_64x128 = (1, 4, 2, 4, 2)
_SHAPE_128x128 = (2, 2, 2, 4, 4)
_SHAPE_128x256 = (1, 8, 4, 8, 2)
_SHAPE_128x256_TALL = (2, 4, 4, 4, 2)

# Weights up to this size stay in L2 across M tiles, so the tile height stops being the throughput and
# the pick follows the output width and the grid instead.
_WEIGHT_L2_SMALL_BYTES = 8 << 20


def tile_shape(tile: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    """`(block_m, block_n, chunk_k)` of a tile."""
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = tile
    return 16 * warps_m * repeat_m, 16 * warps_n * repeat_n, 16 * unroll_k


def select_scaled_mm_config(
    M: int,
    N: int,
    K: int,
    a_dtype: torch.dtype = torch.float16,
    b_dtype: torch.dtype = torch.float8_e5m2,
    out_dtype: torch.dtype = torch.bfloat16,
) -> tuple[int, int, int, int, int]:
    """Static tile choice for one shape.

    Two regimes, from the weight traffic. A tile re-reads its whole B panel once per M tile, so when the
    weight is larger than L2 the tile height is the throughput and the rule buys a tall tile. When the
    weight fits in L2 the passes are already amortized and the output width decides instead. The
    boundaries are measured on the cell table in the fp8 plan, not derived, so re-fit them with
    `bench/benchmark_scaled_mm_hip.py` rather than reasoning about them.
    """
    del a_dtype, b_dtype, out_dtype

    if M <= 16:
        return _SHAPE_16x256
    if M <= 128:
        return _SHAPE_64x64
    if M <= 256:
        return _SHAPE_64x64 if N <= 4096 else _SHAPE_64x128
    if M <= 512:
        # 512 rows is four M tiles at 128 rows, so the tile height is already bought. The output width and
        # the length of the reduction decide instead.
        if N <= 1280:
            return _SHAPE_32x32
        if N <= 2560:
            return _SHAPE_128x256_TALL if K <= 8192 else _SHAPE_64x128
        if K <= 2048:
            return _SHAPE_64x128
        return _SHAPE_128x128 if N <= 6144 else _SHAPE_128x256
    if N <= 1024:
        return _SHAPE_32x128
    if N <= 4096 or N * K <= _WEIGHT_L2_SMALL_BYTES:
        return _SHAPE_64x128
    return _SHAPE_128x256


def _run(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    tile: tuple[int, int, int, int, int],
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b_prepacked.shape[1]), device=a.device, dtype=out_dtype)
    torch.ops.feather_ops.scaled_mm.default(a, b_prepacked, scale, bias, out, *tile)
    return out


def scaled_mm_hip_configured(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
    block_warps_m: int,
    block_warps_n: int,
    unroll_k: int,
    repeat_m: int,
    repeat_n: int,
) -> torch.Tensor:
    """Run one explicit tile. Used by the sweep and by tests. The heuristic is not consulted."""
    return _run(a, b_prepacked, scale, bias, out_dtype, (block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n))


def scaled_mm_hip(
    a: torch.Tensor,
    b_prepacked: torch.Tensor,
    scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """`(a @ b^T) * scale + bias` against the prepacked fp8 weight `b_prepacked`."""
    tile = select_scaled_mm_config(a.shape[0], b_prepacked.shape[1], a.shape[1], a.dtype, b_prepacked.dtype, out_dtype)
    return _run(a, b_prepacked, scale, bias, out_dtype, tile)


def prepack_b_for_scaled_mm(b: torch.Tensor) -> torch.Tensor:
    """`[K, N]` fp8 -> `[K_pad/16, N, 16]`, the layout the kernel indexes.

    `K` is zero-padded to a multiple of 128 (the widest K chunk a tile can ask for), which is what lets
    the kernel run a ragged K without a staging branch. `comfy_feather/quant.py::prepack_transpose` is
    the same permutation for a `[N, K]` weight and pads the same way.
    """
    k, n = b.shape
    k_pad = -(-k // 128) * 128
    if k_pad != k:
        padded = torch.zeros((k_pad, n), device=b.device, dtype=b.dtype)
        padded[:k] = b
        b = padded
    return b.view(k_pad // 16, 16, n).permute(0, 2, 1).contiguous()
