#!/usr/bin/env python3
"""Kernel-only sweep of `feather_ops.int8_scaled_mm` over the shared grid in bench/cells.py.

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

from bench.cells import cells
from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.int8_convrot import int8_scaled_mm, prepack_int8_convrot_weight, select_int8_config
from feather_ops.tiles import INT8_TILES

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
    parser.add_argument("--groups", default="", help="comma separated grid groups (bench/cells.py)")
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
    grid = [(M, N_w, K_w) for _group, M, N_w, K_w in cells(wanted)]

    cases = [build_case(M, N_w, K_w, transposed_b, configs, not args.no_bf16) for transposed_b in directions for M, N_w, K_w in grid]
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
