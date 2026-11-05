#!/usr/bin/env python3
"""Benchmark the 3x3 stride-1 conv2d kernel against the two torch paths.

`torch_im2col` is `F.conv2d` with cudnn/miopen disabled, which is what ComfyUI runs for bf16 (im2col +
hipBLASLt addmm). `torch_miopen` is `F.conv2d` with cudnn enabled (MIOpen's find-db choice, a Winograd
kernel for fp16). `hip <tile>` is this kernel with the weight prepacked once outside the timed region.
The `rel` column is the relative L2 error against a float64 `F.conv2d` of the same inputs.

Usage:
    python -m bench.benchmark_conv3x3.py --dtype bfloat16
    python -m bench.benchmark_conv3x3.py --shapes 384x3456x65536,1152x10368x16384
    python -m bench.benchmark_conv3x3.py --configs 4,4,2,4;3,8,2,2
"""

import argparse

import torch
import torch.nn.functional as F

import feather_ops
from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.conv2d import prepack_conv2d_weight_3x3, select_conv3x3_config
from feather_ops.tiles import CONV_TILES, conv_block_sizes

torch.set_grad_enabled(False)

# (C_out, K = 9 * C_in, N = H * W) for the 2D VAE 3x3 stride-1 shapes
DEFAULT_SHAPES = [
    (512, 4608, 65536),
    (384, 3456, 65536),
    (256, 2304, 65536),
    (192, 1728, 65536),
    (128, 1152, 65536),
    (96, 864, 65536),
    (1152, 10368, 16384),
    (288, 2592, 65536),
    (576, 5184, 65536),
    (144, 1296, 65536),
    (96, 864, 262144),
    (128, 1152, 262144),
    (384, 3456, 16384),
    (1152, 10368, 65536),
]


def _tile_from_text(text: str) -> tuple[int, int, int, int, int]:
    values = [int(value) for value in text.split(",")]
    assert len(values) == 5, f"a tile is five ints (warps_m, warps_n, repeat_m, repeat_n, rows), got {text!r}"
    tile = (values[0], values[1], values[2], values[3], values[4])
    assert tile in CONV_TILES, f"not instantiated: {tile}"
    return tile


def build_case(c_out: int, k: int, n: int, dtype: torch.dtype, tiles: list[tuple[int, int, int, int, int]], variants: str) -> Case:
    c_in = k // 9
    side = round(n**0.5)
    h = w = side
    assert h * w == n, (h, w, n)
    x = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.float32).to(dtype)
    weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    packed = prepack_conv2d_weight_3x3(weight)

    def conv_im2col():
        torch.backends.cudnn.enabled = False
        F.conv2d(x, weight, bias, 1, 1)

    def conv_miopen():
        torch.backends.cudnn.enabled = True
        F.conv2d(x, weight, bias, 1, 1)

    def hip(tile, prefetch=None):
        return lambda: feather_ops.conv3x3_prepacked(x, packed, bias, tile=tile, prefetch=prefetch)

    providers = {"torch_im2col": conv_im2col, "torch_miopen": conv_miopen}
    for tile in tiles:
        name = f"hip {conv_block_sizes(tile)}"
        providers[name] = hip(tile)
        if variants == "both":
            providers[f"{name} prefetch=off"] = hip(tile, False)
            providers[f"{name} prefetch=on"] = hip(tile, True)

    return Case(f"C_out={c_out} C_in={c_in} {h}x{w}", providers, scale=2.0 * c_in * c_out * 9 * h * w / 1e9)


def check(case: Case, c_out: int, k: int, n: int, dtype: torch.dtype) -> None:
    """Print the relative L2 error of every hip provider against a float64 conv2d."""
    c_in = k // 9
    side = round(n**0.5)
    x = torch.randn((1, c_in, side, side), device="cuda", dtype=torch.float32).to(dtype)
    weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    ref = F.conv2d(x.double(), weight.double(), bias.double(), 1, 1)
    packed = prepack_conv2d_weight_3x3(weight)
    for name in case.providers:
        if not name.startswith("hip"):
            continue
        out = feather_ops.conv3x3_prepacked(x, packed, bias)
        rel = float((out.double() - ref).norm() / ref.norm())
        print(f"    {name:26s} rel={rel:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--tiles", default="", help="wm,wn,rm,rn,rows;... (default: the heuristic's tile per shape)")
    ap.add_argument("--variants", choices=["auto", "both"], default="auto", help="'both' adds prefetch=off/on per tile")
    ap.add_argument("--check", action="store_true", help="also print the float64 relative error per shape")
    add_protocol_args(ap)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = [(m, k, n) for m, k, n in (tuple(int(v) for v in token.split("x")) for token in args.shapes.split(","))] if args.shapes else DEFAULT_SHAPES
    explicit = [_tile_from_text(token) for token in args.tiles.split(";")] if args.tiles else None

    cases = []
    for c_out, k, n in shapes:
        if explicit is not None:
            cases.append(build_case(c_out, k, n, dtype, explicit, args.variants))
            continue
        tile = select_conv3x3_config(c_out, round(n**0.5))
        assert tile is not None, f"C_out={c_out} W={round(n**0.5)} has no tile"
        cases.append(build_case(c_out, k, n, dtype, [tile], args.variants))
    if args.check:
        for case, (c_out, k, n) in zip(cases, shapes):
            print(f"== {case.label}")
            check(case, c_out, k, n, dtype)
    records = run_cases(
        cases,
        unit="TFLOPS",
        control="torch_im2col",
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )
    torch.backends.cudnn.enabled = False

    # Totals over the cell table: the sum of the per-case times, which is what a VAE decode pays. The
    # hip row is the best tile per case, since the heuristic picks a different tile per shape.
    totals = {"torch_im2col": 0.0, "torch_miopen": 0.0, "hip (best tile)": 0.0}
    for record in records:
        for name, provider in record["providers"].items():
            if name in ("torch_im2col", "torch_miopen"):
                totals[name] += provider["median_ms"]
        hip = [provider["median_ms"] for name, provider in record["providers"].items() if name.startswith("hip")]
        if hip:
            totals["hip (best tile)"] += min(hip)
    gflop = sum(case.scale for case in cases)
    print(f"\nsum over {len(cases)} shapes:")
    for name, ms in sorted(totals.items(), key=lambda item: item[1]):
        print(f"  {name:22s} {ms:.3f} ms  {gflop / ms:.3f} TFLOPS")


if __name__ == "__main__":
    with torch.inference_mode():
        main()
