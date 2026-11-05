#!/usr/bin/env python3
"""Compare the kernels this repo ships on the shared grid in bench/cells.py.

One cell per row, every requested backend interleaved inside it, so the columns are paired. `bf16` is
`torch.mm` on an already-dequantized weight, i.e. the optimistic control: it does not pay for the
dequantization, and neither kernel is compared against a dequantize-then-GEMM path.

The backends do different arithmetic by design - `fp8` is `bf16/fp16 activations x e5m2 weight`, `int8` is an
`int8 x int8` ConvRot GEMM with per-row and per-channel scales - so read the columns as "how fast is each
kernel on this cell", not as an equivalence. That is the question a router or a quantization choice asks.

Usage:
    python -m bench.benchmark_kernels                              # fp8 fwd, int8 fwd/bwd, bf16
    python -m bench.benchmark_kernels --backends fp8,int8_bf16     # a subset
    python -m bench.benchmark_kernels --groups ladder-narrow,krea2
"""

import argparse

import torch

from bench.cells import cells
from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.int8_convrot import int8_scaled_mm, prepack_int8_weight, select_int8_config
from feather_ops.scaled_mm import prepack_b_for_scaled_mm, scaled_mm_hip

DEV = "cuda"
DTYPE = torch.bfloat16

BACKENDS = {
    "fp8": "feather_ops.scaled_mm_hip, bf16 activations against an fp8e5m2 weight",
    "int8_fwd": "feather_ops.int8_scaled_mm forward: int8 x int8, output K_w",
    "int8_bwd": "feather_ops.int8_scaled_mm backward: int8 x int8, output N_w",
    "bf16": "torch.mm on a dequantized bf16 weight",
}


def build_case(group: str, M: int, N_w: int, K_w: int, backends: set[str]) -> Case:
    gen = torch.Generator(device=DEV).manual_seed(0)
    scale = torch.tensor(2.34, device=DEV, dtype=torch.float32)
    providers = {}

    if "fp8" in backends or "bf16" in backends:
        a16 = torch.randn((M, K_w), device=DEV, dtype=torch.float32).to(torch.bfloat16)
        b_kn = torch.randn((K_w, N_w), device=DEV, dtype=torch.float32).to(torch.float8_e5m2)
        b_fp8 = prepack_b_for_scaled_mm(b_kn)
        weight = (b_kn.t().contiguous().to(torch.bfloat16).float() * scale).to(torch.bfloat16)
        if "fp8" in backends:
            providers["fp8"] = lambda: scaled_mm_hip(a16, b_fp8, scale, None, torch.bfloat16)
        if "bf16" in backends:
            providers["bf16"] = lambda: torch.mm(a16, weight.t())

    if "int8_fwd" in backends or "int8_bwd" in backends:
        packed = prepack_int8_weight(torch.randint(-127, 128, (N_w, K_w), device=DEV, dtype=torch.int8, generator=gen))
        for name, transposed_b in (("int8_fwd", False), ("int8_bwd", True)):
            if name not in backends:
                continue
            contraction, n_out = (N_w, K_w) if transposed_b else (K_w, N_w)
            a8 = torch.randint(-127, 128, (M, contraction), device=DEV, dtype=torch.int8, generator=gen)
            a_scale = torch.rand(M, device=DEV, dtype=torch.float32, generator=gen) * 0.02 + 1e-4
            b_scale = None if transposed_b else torch.rand(n_out, device=DEV, dtype=torch.float32, generator=gen) * 0.02 + 1e-4
            tile = select_int8_config(M, N_w, K_w, transposed_b)
            providers[name] = lambda a8=a8, a_scale=a_scale, b_scale=b_scale, transposed_b=transposed_b, tile=tile: int8_scaled_mm(
                a8, packed, a_scale, b_scale, None, DTYPE, transposed_b, tile
            )

    gflop = 2.0 * M * N_w * K_w / 1e9
    return Case(f"{group} {M}x{N_w}x{K_w}", providers, scale=gflop)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="fp8,int8_fwd,int8_bwd,bf16", help=f"comma separated: {', '.join(BACKENDS)}")
    parser.add_argument("--groups", default="", help="comma separated grid groups (bench/cells.py)")
    add_protocol_args(parser)
    args = parser.parse_args()

    wanted = {value.strip() for value in args.backends.split(",") if value}
    unknown = wanted - set(BACKENDS)
    assert not unknown, f"unknown backend(s) {sorted(unknown)}. Known: {sorted(BACKENDS)}"
    groups = {value.strip() for value in args.groups.split(",") if value} or None

    cases = [build_case(group, m, n, k, wanted) for group, m, n, k in cells(groups)]
    run_cases(
        cases,
        unit="TFLOPS",
        control="bf16" if "bf16" in wanted else None,
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
