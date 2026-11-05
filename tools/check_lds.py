#!/usr/bin/env python3
"""Check each instantiation's real LDS allocation against the size its source asks for.

The kernel's shared storage is a union whose size is a compile-time function of the tile, but the number the
runtime allocates comes from the code object's `group_segment_fixed_size` metadata, and the two can disagree:
`(2,2,2,2,2)` with `prefetch` and fp16 activations was allocated 55296 B against a 6144 B union (that is
what found the `-disable-promote-alloca-to-lds` flag in `tools/hip_flags.py`). A mismatch costs occupancy, so
it is always a bug, either in the source or in the compiler.

Usage: `python tools/check_lds.py` (needs hipcc and ~40 s), or `python tools/check_lds.py --kernel path.cu`.
It is not part of the test suite because it compiles the kernel.
"""

import argparse
import pathlib
import re
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from hip_flags import DEVICE_FLAGS

ROOT = pathlib.Path(__file__).resolve().parents[1]
WMMA_K = 16
WMMA_M = 16
K1 = 8  # 8 halfwords per lane, i.e. the 16-byte vector the A staging moves


def expected_bytes(tile: tuple[int, ...]) -> int:
    """The `SharedStorage` union size of one `scaled_mm_kernel` instantiation."""
    warps_m, warps_n, unroll_k, repeat_m, repeat_n, _prefetch = tile
    block_m = WMMA_M * warps_m * repeat_m
    block_n = WMMA_M * warps_n * repeat_n
    k0 = WMMA_K // K1
    a_bytes = unroll_k * k0 * block_m * K1 * 2
    b_bytes = unroll_k * block_n * WMMA_K
    c_bytes = warps_m * warps_n * WMMA_M * WMMA_K * 2  # the C-shuffle epilogue shares the same union
    return max(a_bytes + b_bytes, c_bytes)


def compile_isa(kernel: pathlib.Path) -> str:
    hipcc = shutil.which("hipcc") or str(pathlib.Path(sys.executable).parent / "hipcc")
    command = [
        hipcc,
        "-O3",
        "--offload-arch=gfx1151",
        "-DNO_PYTORCH",
        "-I",
        str(kernel.parent),
        *DEVICE_FLAGS,
        "--cuda-device-only",
        "-S",
        "-o",
        "-",
        str(kernel),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise SystemExit(f"hipcc failed:\n{result.stderr[-4000:]}")
    return result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", default=str(ROOT / "csrc" / "hip_kernel.cu"))
    args = parser.parse_args()

    text = compile_isa(pathlib.Path(args.kernel))
    # The mangled name lists the tile's five ints then the five bools: `...kernelILi1ELi1ELi2ELi1ELi2ELb1E...`.
    pattern = re.compile(r"scaled_mm_kernelI((?:Li\d+E)+)((?:Lb[01]E)+)")
    current, checked, bad = None, 0, []
    for line in text.splitlines():
        section = re.match(r"\s*\.section\s+\.text\.(\S+),", line)
        if section:
            current = section.group(1)
            continue
        if current is None or "amdhsa_group_segment_fixed_size" not in line:
            continue
        match = pattern.search(current)
        if not match:
            continue
        ints = [int(value) for value in re.findall(r"Li(\d+)E", match.group(1))]
        bools = [int(value) for value in re.findall(r"Lb([01])E", match.group(2))]
        values = bools
        tile = tuple(ints[:5]) + (bools[4],)
        current = None  # one metadata block per kernel
        allocated = int(line.split()[-1])
        expected = expected_bytes(tile)
        checked += 1
        if allocated != expected:
            bad.append((tile, (values[1], values[2], values[3]), allocated, expected))

    print(f"{checked} scaled_mm instantiations checked against their LDS union")
    if not bad:
        print("every allocation matches")
        return 0
    print(f"{len(bad)} mismatches (tile, (a_bf16, out_bf16, tail_free), allocated, expected):")
    for tile, flags, allocated, expected in sorted(set(bad)):
        print(f"  {tile} flags={flags} allocated={allocated} expected={expected}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
