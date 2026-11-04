#!/usr/bin/env python3
"""Benchmark the 3x3 stride-1 conv2d kernel against the two torch paths.

`torch` is `F.conv2d` with cudnn/miopen disabled, which is what ComfyUI runs today (im2col + hipBLASLt addmm).
`torch_miopen` is `F.conv2d` with cudnn enabled (MIOpen's find-db choice).
`hip` is this kernel, with the weight prepacked once outside the timed region.

Usage:
    python bench/benchmark_conv3x3.py --dtype bfloat16
    python bench/benchmark_conv3x3.py --shapes 384x3456x65536,1152x10368x16384
    python bench/benchmark_conv3x3.py --configs 4,4,2,4;3,8,2,2
"""

import argparse
import statistics

import torch
import torch.nn.functional as F

import feather_ops
from feather_ops.conv2d import SUPPORTED_TILES, _select_config, prepack_conv2d_weight_3x3

torch.set_grad_enabled(False)

# (C_out, K = 9 * C_in, N = H * W) for the 2D VAE 3x3 stride-1 shapes, see
# ~/ComfyUI-FeatherOps/docs/input_shapes_conv2d_vae.md.
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


def timeit(fn, warmup=5, iters=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def timeit_interleaved(fns, rounds=3, warmup=3, iters=12):
    """Median over rounds of every candidate, so the sustained-clock drift hits all of them alike.

    The GPU's clock is a function of what ran before (up to 8 % on this machine), so timing one
    candidate to completion and then the next biases everything measured later.
    """
    for fn in fns.values():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    per_round = {name: [] for name in fns}
    for _ in range(rounds):
        for name, fn in fns.items():
            samples = []
            for _ in range(iters):
                start.record()
                fn()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end))
            per_round[name].append(statistics.median(samples))
    return {name: statistics.median(v) for name, v in per_round.items()}


def run_shape(c_out, k, n, dtype, iters, configs):
    c_in = k // 9
    side = round(n**0.5)
    h = w = side
    assert h * w == n, (h, w, n)
    x = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.float32).to(dtype)
    weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(dtype)
    gflop = 2.0 * c_in * c_out * 9 * h * w / 1e9
    ref = F.conv2d(x.double(), weight.double(), bias.double(), 1, 1)

    def conv_im2col():
        torch.backends.cudnn.enabled = False
        F.conv2d(x, weight, bias, 1, 1)

    def conv_miopen():
        torch.backends.cudnn.enabled = True
        F.conv2d(x, weight, bias, 1, 1)

    packed = prepack_conv2d_weight_3x3(weight)
    auto = _select_config(c_out, w)
    print(f"\n== C_out={c_out} C_in={c_in} K={k} {h}x{w}  ({gflop:.3f} GFLOP, {dtype})  auto {auto}")
    candidates = []
    if configs is None:
        candidates = [auto] if auto else []
    else:
        candidates = [(*c, 1) for c in configs if c in SUPPORTED_TILES]
        if auto and auto not in candidates:
            candidates.append(auto)
    fns = {"im2col": conv_im2col, "miopen": conv_miopen}
    for block_m, block_n, rows in candidates:
        fns[f"hip {block_m}x{block_n}x{rows}"] = lambda bm=block_m, bn=block_n, r=rows: feather_ops.conv3x3_prepacked(x, packed, bias, bm, bn, rows=r)
    times = timeit_interleaved(fns, warmup=3, iters=iters)
    torch.backends.cudnn.enabled = False
    t_torch = times["im2col"]
    t_miopen = times["miopen"]

    print(f"  torch im2col+addmm : {t_torch:.3f} ms  {gflop / t_torch:.3f} TF   x1.00")
    print(f"  torch MIOpen       : {t_miopen:.3f} ms  {gflop / t_miopen:.3f} TF   x{t_torch / t_miopen:.3f}")
    results = []
    for block_m, block_n, rows in candidates:
        out = feather_ops.conv3x3_prepacked(x, packed, bias, block_m, block_n, rows=rows)
        rel = float((out.double() - ref).norm() / ref.norm())
        t = times[f"hip {block_m}x{block_n}x{rows}"]
        results.append((t, block_m, block_n, rows, rel))
    for t, block_m, block_n, rows, rel in sorted(results):
        print(f"  hip tile {block_m:3d}x{block_n:<3d}x{rows} : {t:.3f} ms  {gflop / t:.3f} TF   x{t_torch / t:.3f} vs im2col  x{t_miopen / t:.3f} vs miopen  rel={rel:.1e}")
    return gflop, t_torch, t_miopen, min(results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--configs", default="")
    ap.add_argument("--iters", type=int, default=15)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    shapes = DEFAULT_SHAPES
    if args.shapes:
        shapes = []
        for token in args.shapes.split(","):
            m, k, n = (int(v) for v in token.split("x"))
            shapes.append((m, k, n))
    configs = None
    if args.configs:
        configs = [tuple(int(v) for v in token.split(",")) for token in args.configs.split(";")]

    rows = []
    for c_out, k, n in shapes:
        rows.append(run_shape(c_out, k, n, dtype, args.iters, configs))

    total_flop = sum(r[0] for r in rows)
    t_torch = sum(r[1] for r in rows)
    t_miopen = sum(r[2] for r in rows)
    t_hip = sum(r[3][0] for r in rows)
    print(
        f"\nTOTAL over {len(rows)} shapes: im2col {t_torch:.3f} ms ({total_flop / t_torch:.3f} TF), "
        f"MIOpen {t_miopen:.3f} ms ({total_flop / t_miopen:.3f} TF), "
        f"hip {t_hip:.3f} ms ({total_flop / t_hip:.3f} TF) -> "
        f"x{t_torch / t_hip:.3f} vs im2col, x{t_miopen / t_hip:.3f} vs MIOpen"
    )


if __name__ == "__main__":
    main()
