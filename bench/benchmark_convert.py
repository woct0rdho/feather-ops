#!/usr/bin/env python3
"""Benchmark the bf16 -> fp16 Triton conversion against torch and torch.compile.

Reported as GB/s of traffic (read bf16 + write fp16 = 4 bytes per element).

Usage:
    python -m bench.benchmark_convert.py
    python -m bench.benchmark_convert.py --sizes 1024,4096 --cold
"""

import argparse

import torch

from bench.protocol import Case, add_protocol_args, run_cases
from feather_ops.convert import bf16_to_fp16


def torch_convert(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16)


torch_convert_compiled = torch.compile(torch_convert, fullgraph=True, dynamic=False, mode="max-autotune")


def build_case(n: int) -> Case:
    x = torch.randn((n, n), device="cuda", dtype=torch.bfloat16)
    providers = {
        "torch": lambda: torch_convert(x),
        "torch_compiled": lambda: torch_convert_compiled(x),
        "triton_asm": lambda: bf16_to_fp16(x),
    }
    return Case(f"{n}x{n}", providers, scale=4.0 * n * n / 1e6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", default="1024,2048,4096,8192,16384")
    add_protocol_args(parser)
    args = parser.parse_args()
    run_cases(
        [build_case(int(n)) for n in args.sizes.split(",")],
        unit="GB/s",
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
