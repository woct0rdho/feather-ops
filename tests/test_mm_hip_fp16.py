import pytest
import torch

from feather_ops.mm_fp16 import _CONFIGS, mm_hip_fp16_configured, prepack_b_for_mm_fp16
from feather_ops.naive import scaled_mm_naive

TEST_SIZES = [
    (128, 128, 128),
    (256, 256, 256),
    (256, 256, 512),
    (256, 512, 256),
    (512, 256, 256),
    (512, 512, 512),
    (1024, 1024, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 8192),
]
RTOL = 0.01
ATOL = 1.0


def _block_sizes(cfg: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    return 16 * warps_m * repeat_m, 16 * warps_n * repeat_n, 16 * unroll_k


def _shape_compatible(cfg: tuple[int, int, int, int, int], shape: tuple[int, int, int]) -> bool:
    block_m, block_n, chunk_k = _block_sizes(cfg)
    M, N, K = shape
    return M % block_m == 0 and N % block_n == 0 and K % chunk_k == 0


CASES = [pytest.param(cfg, shape) for cfg in _CONFIGS for shape in TEST_SIZES if _shape_compatible(cfg, shape)]


def _check_config(cfg: tuple[int, int, int, int, int], shape: tuple[int, int, int]) -> tuple[float, float]:
    torch.manual_seed(0)
    M, N, K = shape
    a = torch.randn((M, K), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32).to(torch.float16)
    bias = torch.randn(N, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16

    b_prepacked = prepack_b_for_mm_fp16(b)
    out_hip = mm_hip_fp16_configured(a, b_prepacked, bias, out_dtype, *cfg).float()
    out_ref = scaled_mm_naive(a, b, None, bias, out_dtype).float()

    diff = out_hip - out_ref
    fro_rel_err = (torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(out_ref).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return fro_rel_err, max_abs_err


@pytest.mark.parametrize("cfg,shape", CASES)
def test_config(cfg: tuple[int, int, int, int, int], shape: tuple[int, int, int]) -> None:
    fro_rel_err, max_abs_err = _check_config(cfg, shape)
    assert fro_rel_err <= RTOL, f"fro_rel_err={fro_rel_err:.3g} > {RTOL}"
    assert max_abs_err <= ATOL, f"max_abs_err={max_abs_err:.3g} > {ATOL}"
