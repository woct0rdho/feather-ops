import pytest
import torch

from feather_ops.mm_hipblaslt import mm_hipblaslt_fp16
from feather_ops.naive import scaled_mm_naive

LAYOUTS = ["TT", "TN", "NT", "NN"]
TEST_SIZES = [
    (128, 128, 128),
    (128, 256, 192),
    (256, 128, 192),
    (256, 384, 512),
    (512, 256, 384),
    (512, 512, 512),
    (1024, 1024, 1024),
]
RTOL = 0.01
ATOL = 1.0

CASES = [pytest.param(layout, shape) for layout in LAYOUTS for shape in TEST_SIZES]


def row_major(shape: tuple[int, int]) -> torch.Tensor:
    return torch.randn(shape, device="cuda", dtype=torch.float32).to(torch.float16).contiguous()


def col_major(shape: tuple[int, int]) -> torch.Tensor:
    return torch.randn((shape[1], shape[0]), device="cuda", dtype=torch.float32).to(torch.float16).contiguous().T


def _make_inputs(layout: str, shape: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    M, N, K = shape
    a = row_major((M, K)) if layout[0] == "T" else col_major((M, K))
    b = row_major((K, N)) if layout[1] == "T" else col_major((K, N))
    return a, b


def _check_layout(layout: str, shape: tuple[int, int, int]) -> tuple[float, float]:
    torch.manual_seed(0)
    a, b = _make_inputs(layout, shape)
    M = shape[0]
    scale = torch.rand(M, device="cuda", dtype=torch.float32)
    bias = torch.randn(M, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16

    out_hipblaslt = mm_hipblaslt_fp16(a, b, scale, bias, out_dtype).float()
    out_ref = scaled_mm_naive(a, b, scale, bias, out_dtype, bias_dim=0).float()

    diff = out_hipblaslt - out_ref
    fro_rel_err = (torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(out_ref).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return fro_rel_err, max_abs_err


@pytest.mark.parametrize("layout,shape", CASES)
def test_layout(layout: str, shape: tuple[int, int, int]) -> None:
    fro_rel_err, max_abs_err = _check_layout(layout, shape)
    assert fro_rel_err <= RTOL, f"fro_rel_err={fro_rel_err:.3g} > {RTOL}"
    assert max_abs_err <= ATOL, f"max_abs_err={max_abs_err:.3g} > {ATOL}"
