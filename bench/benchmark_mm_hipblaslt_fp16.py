#!/usr/bin/env python3
"""Benchmark the hipBLASLt fp16 matmul wrapper against torch.mm and F.linear in all four layouts.

TT/TN/NT/NN are the layouts of (a, b) as the operator sees them. `torch` rows map each to the layout
PyTorch's BLAS call actually uses for a row-major output.

Usage:
    python -m bench.benchmark_mm_hipblaslt_fp16.py
    python -m bench.benchmark_mm_hipblaslt_fp16.py --sizes 2048,4096 --layouts TT,NN
"""

import argparse

import torch
import torch.nn.functional as F

from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.mm_hipblaslt import mm_hipblaslt_fp16

# PyTorch swaps operands for a row-major output.
TORCH_LAYOUT = {"TT": "NN", "TN": "TN", "NT": "NT", "NN": "TT"}


def build_case(n: int, layouts: list[str]) -> Case:
    a = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    scale = torch.ones(n, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16

    providers = {}
    for layout in layouts:
        torch_layout = TORCH_LAYOUT[layout]
        a_t, b_t = a, b
        if torch_layout in ("TN", "NN"):
            b_t = b.T
        if torch_layout in ("NT", "NN"):
            a_t = a.T
        providers[f"torch_mm_{layout}"] = lambda a_t=a_t, b_t=b_t: torch.mm(a_t, b_t)
        providers[f"torch_linear_{layout}"] = lambda a_t=a_t, b_t=b_t: F.linear(a_t, b_t.T, bias)

        a_h, b_h = a, b
        if layout in ("TN", "NN"):
            b_h = b.T
        if layout in ("NT", "NN"):
            a_h = a.T
        providers[f"hipblaslt_{layout}"] = lambda a_h=a_h, b_h=b_h: mm_hipblaslt_fp16(a_h, b_h, scale, bias, out_dtype, solution_index=-2)

    return Case(f"{n}x{n}x{n}", providers, scale=2.0 * n**3 / 1e9, control=f"torch_mm_{layouts[0]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096,8192")
    parser.add_argument("--layouts", default="TT,TN,NT,NN")
    add_protocol_args(parser)
    args = parser.parse_args()
    layouts = [value.strip() for value in args.layouts.split(",")]
    run_cases(
        [build_case(int(n), layouts) for n in args.sizes.split(",")],
        unit="TFLOPS",
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
