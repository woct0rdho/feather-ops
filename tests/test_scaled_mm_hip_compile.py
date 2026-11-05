import pytest
import torch

from feather_ops.naive import scaled_mm_naive
from feather_ops.scaled_mm import prepack_b_for_scaled_mm, scaled_mm_hip

SHAPES = [(4112, 1152, 1152), (512, 1024, 5120)]


def _inputs(M, N, K, a_dtype=torch.bfloat16, out_dtype=torch.bfloat16):
    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float32).to(a_dtype)
    b = torch.randn((K, N), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = torch.randn(N, device="cuda", dtype=out_dtype)
    return a, prepack_b_for_scaled_mm(b), scale, bias


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("shape", SHAPES)
def test_compile_matches_eager(dynamic, shape):
    M, N, K = shape
    a, b_prepacked, scale, bias = _inputs(M, N, K)

    def run(a, b_prepacked, scale, bias):
        return scaled_mm_hip(a, b_prepacked, scale, bias, torch.bfloat16)

    eager = run(a, b_prepacked, scale, bias)
    compiled = torch.compile(run, fullgraph=True, dynamic=dynamic)(a, b_prepacked, scale, bias)
    ref = scaled_mm_naive(a, b_prepacked, scale, bias, torch.bfloat16, b_prepacked=True)
    assert torch.allclose(compiled, eager, atol=1e-2, rtol=1e-2)
    rel = ((compiled.float() - ref.float()).norm() / ref.float().norm()).item()
    assert rel <= 0.01, f"rel={rel:.3g}"
