#!/usr/bin/env python3
"""Kernel-only sweep of feather_ops.int8_scaled_mm over the model cells of the int8 ConvRot plan.

Every cell is (M, N_w, K_w) where N_w/K_w are the weight's output/input axes. The forward contracts K_w
(out = N_w) and the backward contracts N_w (out = K_w), so one cell covers both directions of the same
weight. Configs are interleaved inside one process and reported as medians, per the noise guidance in
~/evotensile/docs/noisy_measurements.md.

Usage:
    python bench/benchmark_int8_convrot.py                # full sweep
    python bench/benchmark_int8_convrot.py --dir fwd      # one direction
    python bench/benchmark_int8_convrot.py --configs 0,3  # subset of the config table
    python bench/benchmark_int8_convrot.py --rounds 5     # more samples
"""

import argparse
import statistics

import torch
import triton

from feather_ops.int8_convrot import INT8_CONFIGS, int8_scaled_mm, prepack_int8_convrot_weight, select_int8_config

# (name, [(M, N_w, K_w), ...])
CELLS = [
    ("krea2", [(4112, 6144, 6144), (4112, 1536, 6144), (4112, 16384, 6144), (4112, 6144, 16384), (9232, 6144, 6144), (9232, 1536, 6144), (9232, 16384, 6144)]),
    ("qwen", [(4112, 4096, 4096), (4112, 24576, 4096), (4112, 4096, 12288), (9232, 4096, 4096)]),
    ("zimage", [(4128, 11520, 3840), (4128, 3840, 3840), (4128, 10240, 3840), (4128, 3840, 10240)]),
    ("h3", [(5376, 21504, 5376), (5376, 5376, 7168), (5376, 28672, 5376), (5376, 5376, 14336), (25344, 21504, 5376)]),
    ("moss", [(4112, 9216, 3072), (4112, 3072, 3072), (4112, 16384, 3072), (4112, 3072, 8192)]),
    ("klein", [(4096, 12288, 4096), (4096, 4096, 4096), (4096, 24576, 4096), (4096, 4096, 12288), (4608, 36864, 4096), (4608, 4096, 16384), (9728, 4096, 4096)]),
    ("anima", [(4096, 2048, 2048), (512, 2048, 1024), (4096, 8192, 2048), (4096, 2048, 8192), (9216, 2048, 2048), (9216, 8192, 2048)]),
    ("text-enc", [(512, 12288, 4096), (1024, 4096, 4096), (2048, 1152, 1152), (1024, 3456, 1152), (2048, 4304, 1152), (2048, 1152, 4304), (512, 248320, 4096)]),
    ("path", [(1, 36864, 6144), (16, 6144, 6144), (192, 2560, 2560), (512, 4096, 4096), (2048, 6144, 6144), (3072, 6144, 6144), (4096, 3456, 1152)]),
]

DEV = "cuda"
DTYPE = torch.bfloat16


def make_inputs(M, N_w, K_w, seed=0):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    wq = torch.randint(-127, 128, (N_w, K_w), device=DEV, dtype=torch.int8, generator=gen)
    a_scale = torch.rand(M, device=DEV, dtype=torch.float32, generator=gen) * 0.02 + 1e-4
    return prepack_int8_convrot_weight(wq), a_scale


def time_cell(M, N_w, K_w, config, transposed_b, rounds):
    contraction, n_out = (N_w, K_w) if transposed_b else (K_w, N_w)
    packed, a_scale = make_inputs(M, N_w, K_w)
    a = torch.randint(-127, 128, (M, contraction), device=DEV, dtype=torch.int8).contiguous()
    b_scale = None if transposed_b else (torch.rand(n_out, device=DEV, dtype=torch.float32) * 0.02 + 1e-4)
    times = []
    for _ in range(rounds):
        fn = lambda: int8_scaled_mm(a, packed, a_scale, b_scale, None, DTYPE, transposed_b, config)
        times.append(triton.testing.do_bench(fn, warmup=20, rep=60))
    tflops = 2 * M * n_out * contraction / (statistics.median(times) * 1e9)
    return statistics.median(times), tflops


def bf16_reference(M, N_w, K_w, transposed_b):
    """What the int8 kernel replaces: a bf16 GEMM on the same shapes."""
    contraction, n_out = (N_w, K_w) if transposed_b else (K_w, N_w)
    a = torch.randn(M, contraction, device=DEV, dtype=DTYPE)
    w = torch.randn(contraction, n_out, device=DEV, dtype=DTYPE)
    ms = triton.testing.do_bench(lambda: a @ w, warmup=20, rep=60)
    return ms, 2 * M * n_out * contraction / (ms * 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", choices=["fwd", "bwd", "both"], default="both")
    parser.add_argument("--configs", default="")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--groups", default="")
    parser.add_argument("--bf16", action="store_true", help="also time the bf16 GEMM it replaces")
    parser.add_argument("--verbose", action="store_true", help="print every config for every cell")
    parser.add_argument("--heuristic", action="store_true", help="time only the schedule select_int8_config picks per cell")

    args = parser.parse_args()

    configs = [INT8_CONFIGS[int(i)] for i in args.configs.split(",")] if args.configs else list(INT8_CONFIGS)
    dirs = [False, True] if args.dir == "both" else [args.dir == "bwd"]
    groups = [g for g in CELLS if not args.groups or g[0] in args.groups.split(",")]

    print(f"# configs={len(configs)} rounds={args.rounds}")
    for cfg in configs:
        wm, wn, uk, rm, rn = cfg
        print(f"# cfg {cfg} -> block {16 * wm * rm:3d}x{16 * wn * rn:3d} chunk_k {16 * uk:3d} threads {32 * wm * wn}")
    print(
        f"{'model':10s} {'dir':4s} {'M':>6s} {'N':>6s} {'K':>6s} {'best cfg':22s} {'ms':>8s} {'TFLOPS':>7s} {'2nd':>22s} {'gap':>6s}"
        + ("  bf16_TF  int8/bf16" if args.bf16 else "")
    )

    for name, cells in groups:
        for M, N_w, K_w in cells:
            for transposed_b in dirs:
                results = {}
                cell_configs = [select_int8_config(M, N_w, K_w, transposed_b)] if args.heuristic else configs
                for cfg in cell_configs:
                    ms, tflops = time_cell(M, N_w, K_w, cfg, transposed_b, args.rounds)
                    results[cfg] = (ms, tflops)
                ranked = sorted(results.items(), key=lambda kv: kv[1][0])
                if args.verbose:
                    detail = "  ".join(f"{cfg[0]}x{cfg[1]}r{cfg[3]}x{cfg[4]}u{cfg[2]}={tf:5.1f}" for cfg, (_ms, tf) in ranked)
                    print(f"# {name} {'bwd' if transposed_b else 'fwd'} {M}x{N_w}x{K_w}: {detail}", flush=True)
                best_cfg, (best_ms, best_tf) = ranked[0]
                if len(ranked) > 1:
                    second_cfg, (second_ms, _) = ranked[1]
                    second_txt = f"{second_cfg!s:22s} {second_ms / best_ms:6.3f}"
                else:
                    second_txt = f"{'-':22s} {'-':>6s}"
                line = f"{name:10s} {'bwd' if transposed_b else 'fwd':4s} {M:6d} {N_w:6d} {K_w:6d} {best_cfg!s:22s} {best_ms:8.3f} {best_tf:7.1f} {second_txt}"
                if args.bf16:
                    bf_ms, bf_tf = bf16_reference(M, N_w, K_w, transposed_b)
                    line += f"  {bf_tf:7.1f}  {bf_ms / best_ms:6.2f}x"
                print(line, flush=True)


if __name__ == "__main__":
    main()
