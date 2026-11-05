#!/usr/bin/env python3
"""Kernel-only benchmark of the fp16 x fp8 prepacked matmul on the shared grid in bench/cells.py.

`torch_mm` is `torch.mm` on an already-dequantized bf16 weight, i.e. the optimistic control: it does
not pay for the dequantization. `hip` is the public entry point, which picks the tile from the static
heuristic. `--tiles` replaces it with an explicit list so the tile sweep and the heuristic can be
compared inside one run.

Usage:
    python -m bench.benchmark_scaled_mm_hip.py                    # the whole grid, heuristic vs torch.mm
    python -m bench.benchmark_scaled_mm_hip.py --groups ladder-wide,krea2
    python -m bench.benchmark_scaled_mm_hip.py --tiles            # add every tile as a provider
    python -m bench.benchmark_scaled_mm_hip.py --tiles 1,8,4,8,2  # one explicit tile
    python -m bench.benchmark_scaled_mm_hip.py --square           # the N^3 sweep, as a regression guard
    python -m bench.benchmark_scaled_mm_hip.py --cells "4112,6144,6144;512,2048,2048"
"""

import argparse

import torch

from bench.cells import cells
from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.scaled_mm import prepack_b_for_scaled_mm, scaled_mm_hip
from feather_ops.tiles import SCALED_MM_TILES


def parse_tiles(text: str) -> list[tuple[int, int, int, int, int, int]]:
    if text == "":
        return list(SCALED_MM_TILES)
    tiles: list[tuple[int, int, int, int, int, int]] = []
    for chunk in text.split(";"):
        values = [int(v) for v in chunk.split(",")]
        assert len(values) == 6, f"a tile is six ints (warps_m, warps_n, unroll_k, repeat_m, repeat_n, prefetch), got {chunk!r}"
        tile = (values[0], values[1], values[2], values[3], values[4], values[5])
        assert tile in SCALED_MM_TILES, f"not instantiated: {tile}"
        tiles.append(tile)
    return tiles


def build_case(family: str, M: int, N: int, K: int, tiles) -> Case:
    a = torch.randn((M, K), device="cuda", dtype=torch.float32).to(torch.bfloat16)
    # The prepack takes the GEMM's B, i.e. [K, N]. The control needs the weight, i.e. [N, K].
    b_kn = torch.randn((K, N), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    b_prepacked = prepack_b_for_scaled_mm(b_kn)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    weight = (b_kn.t().contiguous().to(torch.bfloat16).float() * scale).to(torch.bfloat16)

    providers = {"torch_mm": lambda: torch.mm(a, weight.t())}
    for tile in tiles:
        providers[f"hip {tile}"] = lambda tile=tile: scaled_mm_hip(a, b_prepacked, scale, bias, torch.bfloat16, tile=tile)
    if not tiles:
        providers["hip"] = lambda: scaled_mm_hip(a, b_prepacked, scale, bias, torch.bfloat16)

    gflop = 2.0 * M * N * K / 1e9
    return Case(f"{family} {M}x{N}x{K}", providers, scale=gflop)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", default=None, help="M,N,K;M,N,K override")
    parser.add_argument("--groups", default="", help="comma separated grid groups (bench/cells.py)")
    parser.add_argument("--tiles", nargs="?", const="", default=None, help="add explicit tiles: 'm,n,u,rm,rn,prefetch;...' or nothing for all")
    parser.add_argument("--square", action="store_true", help="run the N^3 sweep instead of the real cells")
    add_protocol_args(parser)
    args = parser.parse_args()

    tiles = parse_tiles(args.tiles) if args.tiles is not None else []
    shapes: list[tuple[int, int, int]] = []
    labels: list[str] = []
    if args.square:
        shapes = [(n, n, n) for n in (512, 1024, 2048, 4096, 8192)]
        labels = ["square"] * len(shapes)
    elif args.cells:
        for chunk in args.cells.split(";"):
            m, n, k = (int(v) for v in chunk.split(","))
            shapes.append((m, n, k))
            labels.append("custom")
    else:
        wanted = {value.strip() for value in args.groups.split(",") if value} or None
        for group, m, n, k in cells(wanted):
            shapes.append((m, n, k))
            labels.append(group)

    cases = [build_case(label, m, n, k, tiles) for label, (m, n, k) in zip(labels, shapes)]
    run_cases(
        cases,
        unit="TFLOPS",
        control="torch_mm",
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
