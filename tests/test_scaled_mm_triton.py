import torch

from feather_ops.naive import scaled_mm_naive
from feather_ops.triton.scaled_mm import scaled_mm_triton

RTOL = 0.01
ATOL = 2.0


def _run_case() -> tuple[float, float]:
    torch.manual_seed(0)
    M = 256
    N = 256
    K = 256
    a_dtype = torch.float16
    b_dtype = torch.float8_e4m3fn
    out_dtype = torch.float16

    a = torch.randn((M, K), device="cuda", dtype=torch.float32).to(a_dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32).to(b_dtype)
    scale = torch.tensor(2.34, device="cuda", dtype=out_dtype)
    bias = torch.randn(N, device="cuda", dtype=out_dtype)

    out_triton = scaled_mm_triton(a, b, scale, bias, out_dtype).float()
    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype).float()

    diff = out_triton - out_ref
    fro_rel_err = (torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(out_ref).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return fro_rel_err, max_abs_err


def test_scaled_mm_triton() -> None:
    fro_rel_err, max_abs_err = _run_case()
    assert fro_rel_err <= RTOL, f"fro_rel_err={fro_rel_err:.3g} > {RTOL}"
    assert max_abs_err <= ATOL, f"max_abs_err={max_abs_err:.3g} > {ATOL}"
