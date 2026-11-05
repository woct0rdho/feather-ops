"""The instantiations of the three kernels, and the tile vocabulary the heuristics choose from.

This module is the single source of truth: `tools/gen_tiles.py` compiles these lists into
`csrc/feather_tiles.h`, which is what the `.cu` files use to instantiate and dispatch, so a tile that the
heuristics can return is always an instantiated kernel. Adding a tile is one line here plus a rebuild.
It must not import torch, because `setup.py` reads it before the extension exists.

Tile formats, all `(block_warps_m, block_warps_n, ...)`, i.e. `block_m = 16 * warps_m * repeat_m`:
- `SCALED_MM_TILES` and `INT8_TILES`: `(warps_m, warps_n, unroll_k, repeat_m, repeat_n)`, with
  `chunk_k = 16 * unroll_k`.
- `CONV_TILES`: `(warps_m, warps_n, repeat_m, repeat_n, rows)`, where `rows` is how many output rows one
  block computes. The conv kernel has no `unroll_k`: its k loop is the nine taps.
"""

CONV_TILES = [
    (4, 4, 2, 4, 1),
    (4, 4, 2, 3, 1),
    (4, 4, 2, 2, 1),
    (4, 2, 2, 3, 1),
    (8, 2, 1, 2, 1),
    (8, 1, 1, 2, 1),
    (3, 8, 2, 2, 1),
    (3, 4, 2, 3, 1),
    (3, 4, 2, 2, 1),
    (3, 3, 2, 2, 1),
    (3, 2, 2, 2, 1),
    (3, 1, 2, 2, 1),
    (2, 8, 2, 2, 1),
    (2, 4, 2, 3, 1),
    (2, 4, 2, 2, 1),
    (2, 2, 2, 3, 1),
    (2, 2, 2, 2, 1),
    (4, 1, 1, 2, 1),
    (1, 8, 3, 2, 1),
    (1, 4, 3, 3, 1),
    (1, 4, 3, 2, 1),
    (1, 2, 3, 3, 1),
    (1, 2, 3, 2, 1),
    (1, 1, 3, 2, 1),
    (2, 8, 1, 2, 1),
    (2, 4, 1, 3, 1),
    (2, 4, 1, 2, 1),
    (2, 3, 1, 2, 1),
    (2, 2, 1, 2, 1),
    (2, 1, 1, 2, 1),
    (1, 8, 1, 2, 1),
    (1, 4, 1, 3, 1),
    (1, 4, 1, 2, 1),
    (1, 3, 1, 2, 1),
    (1, 2, 1, 2, 1),
    (1, 1, 1, 2, 1),
    (4, 4, 2, 2, 2),
    (2, 8, 2, 2, 2),
]

INT8_TILES = [
    (1, 1, 2, 1, 2),
    (1, 2, 2, 1, 2),
    (1, 4, 2, 1, 2),
    (1, 8, 2, 1, 2),
    (1, 1, 4, 1, 2),
    (1, 2, 4, 1, 2),
    (1, 4, 4, 1, 2),
    (1, 8, 4, 1, 2),
    (1, 2, 2, 2, 2),
    (1, 4, 2, 2, 2),
    (2, 2, 2, 2, 2),
    (2, 4, 2, 2, 2),
    (1, 4, 2, 4, 2),
    (1, 8, 2, 4, 2),
    (2, 2, 2, 4, 4),
    (2, 4, 2, 4, 4),
    (4, 2, 2, 4, 4),
    (2, 4, 4, 4, 4),
    (2, 2, 4, 4, 2),
    (2, 4, 2, 4, 2),
    (2, 8, 2, 4, 2),
    (4, 4, 2, 4, 4),
]

SCALED_MM_TILES = [
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
    (2, 2, 2, 2, 2),
    (2, 1, 4, 2, 2),
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


def gemm_block_sizes(tile: tuple[int, ...]) -> tuple[int, int, int]:
    """`(block_m, block_n, chunk_k)` of a `SCALED_MM_TILES` / `INT8_TILES` entry."""
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = tile
    return 16 * warps_m * repeat_m, 16 * warps_n * repeat_n, 16 * unroll_k


def conv_block_sizes(tile: tuple[int, ...]) -> tuple[int, int, int]:
    """`(block_m, block_n, rows)` of a `CONV_TILES` entry."""
    warps_m, warps_n, repeat_m, repeat_n, rows = tile
    return 16 * warps_m * repeat_m, 16 * warps_n * repeat_n, rows


MIOPEN_FP16_MAX_INPUT_ELEMS = 1 << 28
CONV_PREFETCH_MIN_C_IN = 128
