import os
from collections.abc import Callable
from typing import Any, TypeVar

import torch
import triton
from torch._inductor.kernel.custom_op import CustomOpConfig
from torch.fx.experimental.symbolic_shapes import optimization_hint

# Config format: (block_warps_m, block_warps_n, unroll_k, repeat_m, repeat_n[, split_k_factor])
AUTOTUNE_CONFIGS = [
    (1, 1, 2, 1, 2),
    (1, 1, 4, 1, 2),
    (1, 2, 2, 1, 2),
    (1, 2, 4, 1, 2),
    (1, 4, 2, 1, 2),
    (1, 4, 4, 1, 2),
    (1, 8, 2, 1, 2),
    (1, 8, 4, 1, 2),
    (1, 1, 2, 2, 2),
    (1, 1, 4, 2, 2),
    (1, 1, 2, 4, 4),
    (1, 1, 4, 4, 4),
    (1, 2, 2, 2, 2),
    (1, 2, 4, 2, 2),
    (2, 1, 2, 2, 2),
    (2, 1, 4, 2, 2),
    (1, 4, 2, 4, 2),
    (1, 4, 4, 4, 2),
    (1, 8, 2, 8, 2),
    (1, 8, 4, 8, 2),
    (2, 2, 2, 4, 4),
    (2, 2, 4, 4, 4),
    (2, 4, 2, 4, 2),
    (2, 4, 4, 4, 2),
    (2, 4, 2, 4, 4),
    (2, 4, 4, 4, 4),
    (4, 2, 2, 2, 4),
    (4, 2, 4, 2, 4),
]
# TODO: Sort configs with a better heuristic to find the fastest one
AUTOTUNE_CONFIGS = sorted(AUTOTUNE_CONFIGS, key=lambda x: (x[0], x[1], x[3], x[4], x[2]))

ConfigT = TypeVar("ConfigT", bound=tuple[int, ...])


def _config_compatible(cfg: tuple[int, ...], M: int, N: int, K: int) -> bool:
    warps_m, warps_n, unroll_k, repeat_m, repeat_n, *rest = cfg
    split_k_factor = rest[0] if rest else 1
    block_m = 16 * warps_m * repeat_m
    block_n = 16 * warps_n * repeat_n
    chunk_k = 16 * unroll_k

    if M % block_m != 0 or N % block_n != 0 or K % chunk_k != 0:
        return False
    if (K // chunk_k) % split_k_factor != 0:
        return False

    is_large = K >= 3072 and min(M, N) >= 3072 and max(M, N) >= 4096
    return not (is_large and (block_m < 64 or block_n < 64))


def _size_hint(value: int) -> int:
    try:
        return int(value)
    except TypeError:
        return optimization_hint(value)


def generate_autotune_configs(
    fake_tensors: dict[str, torch.Tensor],
    configs: list[ConfigT],
    b_dim: int,
) -> list[CustomOpConfig]:
    a = fake_tensors["a"]
    b_prepacked = fake_tensors["b_prepacked"]
    M = _size_hint(a.shape[0])
    N = _size_hint(b_prepacked.shape[b_dim])
    K = _size_hint(a.shape[1])
    compatible = [cfg for cfg in configs if _config_compatible(cfg, M, N, K)]
    if not compatible:
        raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")

    custom_op_configs = []
    for cfg in compatible:
        params: dict[str, Any] = {
            "block_warps_m": cfg[0],
            "block_warps_n": cfg[1],
            "unroll_k": cfg[2],
            "repeat_m": cfg[3],
            "repeat_n": cfg[4],
        }
        # Scaled-MM configs carry a trailing split-K factor. The fp16 kernel's do not.
        if len(cfg) > 5:
            params["split_k_factor"] = cfg[5]
        custom_op_configs.append(CustomOpConfig(**params))
    return custom_op_configs


def get_compatible_config(a: torch.Tensor, b_prepacked: torch.Tensor, b_dim: int, configs: list[ConfigT]) -> ConfigT:
    M = _size_hint(a.shape[0])
    N = _size_hint(b_prepacked.shape[b_dim])
    K = _size_hint(a.shape[1])
    for cfg in configs:
        if _config_compatible(cfg, M, N, K):
            return cfg
    raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")


_AUTOTUNE_CACHE = {}


# torch autotuner introduces overhead, so we use the old autotuner in benchmarks
def old_autotune(
    M: int,
    N: int,
    K: int,
    configs: list[ConfigT],
    run_fn: Callable[[ConfigT], object],
    *extra_keys: object,
) -> ConfigT:
    key = (M, N, K, *extra_keys)
    cached = _AUTOTUNE_CACHE.get(key)
    if cached is not None:
        return cached

    candidates = [c for c in configs if _config_compatible(c, M, N, K)]
    if not candidates:
        raise RuntimeError(f"No compatible config for M={M} N={N} K={K}.")

    # Same defaults as triton.autotune
    warmup_ms = max(1, int(os.environ.get("AUTOTUNE_WARMUP_MS", "25")))
    rep_ms = max(1, int(os.environ.get("AUTOTUNE_REP_MS", "100")))
    best_cfg = candidates[0]
    best_ms = None

    for cfg in candidates:
        ms = triton.testing.do_bench(lambda cfg=cfg: run_fn(cfg), warmup=warmup_ms, rep=rep_ms)
        if best_ms is None or ms < best_ms:
            best_ms = ms
            best_cfg = cfg

    _AUTOTUNE_CACHE[key] = best_cfg
    print(f"autotune key={key} cfg={best_cfg} time={best_ms:.3f} ms")
    return best_cfg
