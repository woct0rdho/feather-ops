#!/usr/bin/env python3
"""Kernel-only sweep of `feather_ops.int8_scaled_mm` over the model cells of the int8 ConvRot plan.

Every cell is (M, N_w, K_w) where N_w/K_w are the weight's output/input axes. The forward contracts K_w
(out = N_w) and the backward contracts N_w (out = K_w), so one cell covers both directions of the same
weight. Each schedule in `INT8_TILES` is a provider, so they are interleaved and paired against each
other inside one run. `--heuristic` replaces them with the single schedule `select_int8_config` picks,
and `--bf16` (on by default) adds the bf16 GEMM this kernel replaces as the control.

Usage:
    python -m bench.benchmark_int8_convrot.py                  # heuristic + bf16 control, every cell
    python -m bench.benchmark_int8_convrot.py --all-configs    # every schedule as a provider
    python -m bench.benchmark_int8_convrot.py --dir fwd --groups krea2,h3
    python -m bench.benchmark_int8_convrot.py --configs 0,3    # a subset of the schedule table
"""

import argparse

import torch

from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.int8_convrot import int8_scaled_mm, prepack_int8_convrot_weight, select_int8_config
from feather_ops.tiles import INT8_TILES

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
    # coverage audit: cells a training step runs that the model rows above do not list
    ("h3-adaln", [(1, 96768, 2688)]),  # adaln_proj.linear, group 64, forward only, M = batch
    ("krea2-musubi", [(2560, 6144, 6144), (4352, 6144, 6144), (4352, 1536, 6144), (4352, 16384, 6144), (4352, 6144, 16384), (9472, 16384, 6144)]),  # M padded to 256
    ("h3-ragged", [(5251, 21504, 5376), (25343, 21504, 5376)]),  # video+audio+text rows are not 256-aligned
    ("text-enc-short", [(32, 25600, 5120), (32, 5120, 25600), (128, 1024, 5120), (128, 8192, 5120)]),  # short prompts
    # the small-M tiers: TE at 192/256 tokens (the 64x64 tile's crossover is on the output width, not M)
    ("text-enc-192", [(192, 2560, 2560), (192, 12288, 4096), (256, 12288, 4096), (256, 4096, 4096)]),
]

DEV = "cuda"
DTYPE = torch.bfloat16


def build_case(M: int, N_w: int, K_w: int, transposed_b: bool, configs, bf16: bool) -> Case:
    contraction, n_out = (N_w, K_w) if transposed_b else (K_w, N_w)
    gen = torch.Generator(device=DEV).manual_seed(0)
    wq = torch.randint(-127, 128, (N_w, K_w), device=DEV, dtype=torch.int8, generator=gen)
    packed = prepack_int8_convrot_weight(wq)
    a = torch.randint(-127, 128, (M, contraction), device=DEV, dtype=torch.int8).contiguous()
    a_scale = torch.rand(M, device=DEV, dtype=torch.float32, generator=gen) * 0.02 + 1e-4
    b_scale = None if transposed_b else (torch.rand(n_out, device=DEV, dtype=torch.float32) * 0.02 + 1e-4)

    providers = {}
    if bf16:
        weight = torch.randn(contraction, n_out, device=DEV, dtype=DTYPE)
        activation = a.to(DTYPE)
        providers["bf16 gemm"] = lambda activation=activation, weight=weight: activation @ weight

    def run(tile):
        return lambda: int8_scaled_mm(a, packed, a_scale, b_scale, None, DTYPE, transposed_b, tile)

    if configs is None:
        picked = select_int8_config(M, n_out, contraction, transposed_b)
        providers["hip (heuristic)"] = run(picked)
    else:
        for index in configs:
            providers[f"hip cfg{index}"] = run(INT8_TILES[index])

    direction = "bwd" if transposed_b else "fwd"
    return Case(f"{direction} {M}x{n_out}x{contraction}", providers, scale=2 * M * n_out * contraction / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", choices=["fwd", "bwd", "both"], default="fwd")
    parser.add_argument("--configs", default="", help="comma separated INT8_TILES indices. Default is the heuristic")
    parser.add_argument("--all-configs", action="store_true", help="every schedule as a provider")
    parser.add_argument("--groups", default="", help="comma separated cell groups")
    parser.add_argument("--no-bf16", action="store_true", help="do not time the bf16 GEMM control")
    add_protocol_args(parser)
    args = parser.parse_args()

    if args.all_configs:
        configs = list(range(len(INT8_TILES)))
    elif args.configs:
        configs = [int(value) for value in args.configs.split(",")]
    else:
        configs = None

    wanted = {value.strip() for value in args.groups.split(",")} if args.groups else None
    directions = {"fwd": [False], "bwd": [True], "both": [False, True]}[args.dir]
    cells = [(M, N_w, K_w) for name, group in CELLS if wanted is None or name in wanted for M, N_w, K_w in group]

    cases = [build_case(M, N_w, K_w, transposed_b, configs, not args.no_bf16) for transposed_b in directions for M, N_w, K_w in cells]
    run_cases(
        cases,
        unit="TFLOPS",
        control="bf16 gemm" if not args.no_bf16 else None,
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
