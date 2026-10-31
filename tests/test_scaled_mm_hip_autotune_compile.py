import pytest
import torch

from feather_ops.naive import scaled_mm_naive
from feather_ops.scaled_mm import _CONFIGS, prepack_b_for_scaled_mm, scaled_mm_hip, scaled_mm_hip_configured

RTOL = 0.01
ATOL = {
    torch.float16: 2.0,
    torch.bfloat16: 16.0,
}
CONFIGURED_CASES = [
    pytest.param(_CONFIGS[0]),
    pytest.param(_CONFIGS[-1]),
]


def _make_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.dtype]:
    torch.manual_seed(0)
    n = 512
    a = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = torch.randn(n, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16
    b_prepacked = prepack_b_for_scaled_mm(b)
    return a, b, b_prepacked, scale, bias, out_dtype


def _check_close(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    out = out.float()
    ref = ref.float()
    diff = out - ref
    fro_rel_err = (torch.linalg.matrix_norm(diff) / torch.linalg.matrix_norm(ref).clamp(min=1e-6)).item()
    max_abs_err = diff.abs().max().item()
    return fro_rel_err, max_abs_err


def _assert_close(out: torch.Tensor, ref: torch.Tensor, out_dtype: torch.dtype) -> None:
    fro_rel_err, max_abs_err = _check_close(out, ref)
    assert fro_rel_err <= RTOL, f"fro_rel_err={fro_rel_err:.3g} > {RTOL}"
    assert max_abs_err <= ATOL[out_dtype], f"max_abs_err={max_abs_err:.3g} > {ATOL[out_dtype]}"


@pytest.mark.parametrize("cfg", CONFIGURED_CASES)
def test_eager_configured(cfg: tuple[int, int, int, int, int]) -> None:
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs()
    out = scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *cfg)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _assert_close(out, ref, out_dtype)


@pytest.mark.parametrize("cfg", CONFIGURED_CASES)
def test_torch_compile_configured(cfg: tuple[int, int, int, int, int]) -> None:
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs()

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *cfg)

    out = compiled_fn(a, b_prepacked, scale, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _assert_close(out, ref, out_dtype)


def test_eager_autotuned() -> None:
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs()
    out = scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _assert_close(out, ref, out_dtype)


def test_torch_compile_autotuned() -> None:
    a, b, b_prepacked, scale, bias, out_dtype = _make_inputs()

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, scale, bias):
        return scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)

    out = compiled_fn(a, b_prepacked, scale, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _assert_close(out, ref, out_dtype)


def test_torch_compile_autotuned_no_scale() -> None:
    a, b, b_prepacked, _, bias, out_dtype = _make_inputs()
    scale = None

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(a, b_prepacked, bias):
        return scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)

    out = compiled_fn(a, b_prepacked, bias)
    ref = scaled_mm_naive(a, b, scale, bias, out_dtype)
    _assert_close(out, ref, out_dtype)


def test_torch_compile_view_input_no_scale() -> None:
    torch.manual_seed(0)
    x = torch.randn((2, 16, 3072), device="cuda", dtype=torch.bfloat16)
    b = torch.randn((3072, 3072), device="cuda", dtype=torch.float32).to(torch.float8_e5m2)
    b_prepacked = prepack_b_for_scaled_mm(b)
    bias = torch.randn(3072, device="cuda", dtype=torch.float16)
    out_dtype = torch.float16

    @torch.compile(fullgraph=True, mode="max-autotune")
    def compiled_fn(x, b_prepacked, bias):
        x = x.to(torch.float16)
        x = x.view(-1, x.shape[-1])
        return scaled_mm_hip(x, b_prepacked, None, bias, out_dtype)

    out = compiled_fn(x, b_prepacked, bias)
    ref = scaled_mm_naive(x.to(torch.float16).view(-1, 3072), b, None, bias, out_dtype)
    _assert_close(out, ref, out_dtype)
