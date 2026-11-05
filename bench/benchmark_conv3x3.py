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
from feather_ops.conv2d import SUPPORTED_TILES, _select_config, prepack_conv2d_weight_3x3

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


def build_case(c_out: int, k: int, n: int, dtype: torch.dtype, configs) -> Case:
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

    providers = {"torch_im2col": conv_im2col, "torch_miopen": conv_miopen}
    auto = _select_config(c_out, w)
    tiles = [auto] if auto else []
    if configs is not None:
        tiles = [(*c, 1) for c in configs if c in SUPPORTED_TILES] + ([auto] if auto and auto not in [(*c, 1) for c in configs] else [])
    for block_m, block_n, rows in tiles:
        providers[f"hip {block_m}x{block_n}x{rows}"] = lambda bm=block_m, bn=block_n, r=rows: feather_ops.conv3x3_prepacked(x, packed, bias, bm, bn, rows=r)

    return Case(f"C_out={c_out} C_in={c_in} {h}x{w}", providers, scale=2.0 * c_in * c_out * 9 * h * w / 1e9)


def check(case: Case, c_out: int, k: int, n: int, dtype: torch.dtype, configs) -> None:
    """Print the relative L2 error of every hip tile against a float64 conv2d."""
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
        block_m, block_n, rows = (int(v) for v in name.removeprefix("hip ").split("x"))
        out = feather_ops.conv3x3_prepacked(x, packed, bias, block_m, block_n, rows=rows)
        rel = float((out.double() - ref).norm() / ref.norm())
        print(f"    {name:16s} rel={rel:.1e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--check", action="store_true", help="also print the float64 relative error per tile")
    add_protocol_args(ap)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = [(m, k, n) for m, k, n in (tuple(int(v) for v in token.split("x")) for token in args.shapes.split(","))] if args.shapes else DEFAULT_SHAPES
    configs = [tuple(int(v) for v in token.split(",")) for token in args.configs.split(";")] if args.configs else None

    cases = [build_case(c_out, k, n, dtype, configs) for c_out, k, n in shapes]
    if args.check:
        for case, (c_out, k, n) in zip(cases, shapes):
            print(f"== {case.label}")
            check(case, c_out, k, n, dtype, configs)
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
