import pytest
import torch

from feather_ops.naive import scaled_mm_naive
from feather_ops.scaled_mm import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip_configured

TEST_SIZES = [
    (16, 128, 128),
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
DTYPE_COMBOS = [
    (torch.float8_e4m3fn, torch.float16),
    (torch.float8_e4m3fn, torch.bfloat16),
    (torch.float8_e5m2, torch.float16),
    (torch.float8_e5m2, torch.bfloat16),
]
RTOL = 0.01
ATOL = {
    torch.float16: 2.0,
    torch.bfloat16: 16.0,
}


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _block_sizes(cfg: tuple[int, int, int, int, int]) -> tuple[int, int, int]:
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    return 16 * warps_m * repeat_m, 16 * warps_n * repeat_n, 16 * unroll_k


def _shape_compatible(cfg: tuple[int, int, int, int, int], shape: tuple[int, int, int]) -> bool:
    block_m, block_n, chunk_k = _block_sizes(cfg)
    M, N, K = shape
    return M % block_m == 0 and N % block_n == 0 and K % chunk_k == 0


CASES = [pytest.param(cfg, shape, b_dtype, out_dtype) for cfg in _CONFIGS for shape in TEST_SIZES if _shape_compatible(cfg, shape) for b_dtype, out_dtype in DTYPE_COMBOS]


def _check_config(
    cfg: tuple[int, int, int, int, int],
    shape: tuple[int, int, int],
    b_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> tuple[float, float]:
    torch.manual_seed(0)
    M, N, K = shape
    a = torch.randn((M, K), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32).to(b_dtype)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=out_dtype)

    b_prepacked = prepack_b_for_scaled_mm(b)
    out_hip = scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *cfg).float()
    out_ref = scaled_mm_naive(a, b_prepacked, scale, bias, out_dtype, b_prepacked=True).float()

    diff = out_hip - out_ref
    fro_rel_err = (torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(out_ref).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return fro_rel_err, max_abs_err


@pytest.mark.parametrize("cfg,shape,b_dtype,out_dtype", CASES)
def test_config(
    cfg: tuple[int, int, int, int, int],
    shape: tuple[int, int, int],
    b_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    fro_rel_err, max_abs_err = _check_config(cfg, shape, b_dtype, out_dtype)
    assert fro_rel_err <= RTOL, f"fro_rel_err={fro_rel_err:.3g} > {RTOL}"
    assert max_abs_err <= ATOL[out_dtype], f"max_abs_err={max_abs_err:.3g} > {ATOL[out_dtype]}"
