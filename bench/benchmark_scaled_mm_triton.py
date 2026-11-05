#!/usr/bin/env python3
"""Benchmark the Triton implementation of the fp8 scaled matmul against torch and torch.compile.

The HIP kernel has its own benchmark (`benchmark_scaled_mm_hip.py`). This one keeps the Triton path and
the two torch reference paths comparable under the same protocol.

Usage:
    python -m bench.benchmark_scaled_mm_triton.py
    python -m bench.benchmark_scaled_mm_triton.py --sizes 1024,4096
"""

import argparse

import torch

from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.naive import scaled_mm_naive
from feather_ops.triton.scaled_mm import scaled_mm_triton

scaled_mm_naive_compiled = torch.compile(scaled_mm_naive, fullgraph=True, dynamic=False, mode="max-autotune")


def build_case(n: int) -> Case:
    a = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16
    providers = {
        "torch": lambda: scaled_mm_naive(a, b, scale, bias, out_dtype),
        "torch_compiled": lambda: scaled_mm_naive_compiled(a, b, scale, bias, out_dtype),
        "triton": lambda: scaled_mm_triton(a, b, scale, bias, out_dtype),
    }
    return Case(f"{n}x{n}x{n}", providers, scale=2.0 * n**3 / 1e9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="256,512,1024,2048,4096,8192")
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
