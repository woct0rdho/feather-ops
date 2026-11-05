#!/usr/bin/env python3
"""Benchmark the fp16 x fp16 prepacked matmul against torch and torch.compile.

Usage:
    python -m bench.benchmark_mm_hip_fp16.py
    python -m bench.benchmark_mm_hip_fp16.py --sizes 4096,8192
"""

import argparse

import torch

from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.mm_fp16 import mm_hip_fp16, prepack_b_for_mm_fp16
from feather_ops.naive import scaled_mm_naive

scaled_mm_naive_compiled = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune")


def build_case(n: int) -> Case:
    a = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    b_prepacked = prepack_b_for_mm_fp16(b)
    out_dtype = torch.float16
    providers = {
        "torch": lambda: scaled_mm_naive(a, b, None, bias, out_dtype),
        "torch_compiled": lambda: scaled_mm_naive_compiled(a, b, None, bias, out_dtype),
        "hip": lambda: mm_hip_fp16(a, b_prepacked, bias, out_dtype),
    }
    return Case(f"{n}x{n}x{n}", providers, scale=2.0 * n**3 / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="128,256,512,1024,2048,4096,8192")
    add_protocol_args(parser)
    args = parser.parse_args()
    run_cases(
        [build_case(int(n)) for n in args.sizes.split(",")],
        unit="TFLOPS",
        control="torch",
        warmup=args.warmup,
        rounds=args.rounds,
        target_ms=args.target_ms,
        cold=args.cold,
        json_path=args.json_path,
    )


if __name__ == "__main__":
    with torch.inference_mode():
        main()
