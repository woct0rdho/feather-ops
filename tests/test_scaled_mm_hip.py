import pytest
import torch

from feather_ops import scaled_mm
from feather_ops.naive import scaled_mm_naive
from feather_ops.scaled_mm import SCALED_MM_TILES, prepack_b_for_scaled_mm, scaled_mm_hip, scaled_mm_hip_configured

RTOL = 0.01
ATOL = {
    torch.float16: 2.0,
    torch.bfloat16: 16.0,
}

# Model cells (forward) plus the tail shapes the kernel now owns: odd M, M smaller than a tile, and
# N/K that are 16-aligned but not tile-aligned.
SHAPES = [
    (1, 128, 128),
    (17, 96, 64),
    (128, 128, 128),
    (256, 256, 256),
    (512, 1024, 5120),
    (4111, 512, 256),
    (4112, 1152, 1152),
    (4304, 4304, 1152),
    (5376, 512, 3072),
    (2048, 1024, 5120),
]

# One shape with a tail in every axis, cheap enough to sweep the whole tile list.
TAIL_SHAPE = (67, 272, 304)

DTYPES = [(torch.float8_e5m2, torch.float16), (torch.float8_e5m2, torch.bfloat16), (torch.float8_e4m3fn, torch.bfloat16)]


def _tensor(shape, dtype, seed=0):
    torch.manual_seed(seed)
    return torch.randn(shape, device="cuda", dtype=torch.float32).to(dtype)


def _check(a, b_prepacked, scale, bias, out_dtype, tile=None):
    if tile is None:
        out = scaled_mm_hip(a, b_prepacked, scale, bias, out_dtype)
    else:
        out = scaled_mm_hip_configured(a, b_prepacked, scale, bias, out_dtype, *tile)
    ref = scaled_mm_naive(a, b_prepacked, scale, bias, out_dtype, b_prepacked=True)
    diff = (out.float() - ref.float()).abs()
    rel = (diff.norm() / ref.float().norm().clamp(min=1e-6)).item()
    return rel, diff.max().item()


@pytest.mark.parametrize("a_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("b_dtype,out_dtype", DTYPES)
@pytest.mark.parametrize("shape", SHAPES)
def test_shapes(a_dtype, b_dtype, out_dtype, shape):
    M, N, K = shape
    a = _tensor((M, K), a_dtype)
    b = _tensor((K, N), b_dtype)
    scale = torch.tensor(2.34, device="cuda", dtype=torch.float32)
    bias = _tensor((N,), out_dtype)
    rel, max_abs = _check(a, prepack_b_for_scaled_mm(b), scale, bias, out_dtype)
    assert rel <= RTOL, f"rel={rel:.3g}"
    assert max_abs <= ATOL[out_dtype], f"max_abs={max_abs:.3g}"


@pytest.mark.parametrize("tile", SCALED_MM_TILES)
def test_every_tile(tile):
    M, N, K = TAIL_SHAPE
    a = _tensor((M, K), torch.float16)
    b = _tensor((K, N), torch.float8_e5m2)
    scale = torch.tensor(0.75, device="cuda", dtype=torch.float32)
    bias = _tensor((N,), torch.bfloat16)
    rel, max_abs = _check(a, prepack_b_for_scaled_mm(b), scale, bias, torch.bfloat16, tile)
    assert rel <= RTOL, f"rel={rel:.3g}"
    assert max_abs <= ATOL[torch.bfloat16], f"max_abs={max_abs:.3g}"


def test_scale_and_bias_are_optional():
    a = _tensor((96, 64), torch.float16)
    b_prepacked = prepack_b_for_scaled_mm(_tensor((64, 128), torch.float8_e5m2))
    rel, max_abs = _check(a, b_prepacked, None, None, torch.float16)
    assert rel <= RTOL
    assert max_abs <= ATOL[torch.float16]


def test_heuristic_matches_a_measured_tile():
    """The heuristic must return a tile that is actually instantiated and correct for the shape."""
    a = _tensor((4112, 1152), torch.bfloat16)
    b_prepacked = prepack_b_for_scaled_mm(_tensor((1152, 1152), torch.float8_e5m2))
    scale = torch.tensor(1.5, device="cuda", dtype=torch.float32)
    rel, max_abs = _check(a, b_prepacked, scale, None, torch.bfloat16)
    assert rel <= RTOL
    assert max_abs <= ATOL[torch.bfloat16]


def test_heuristic_shapes_are_instantiated():
    for name in dir(scaled_mm):
        if name.startswith("_SHAPE_"):
            assert getattr(scaled_mm, name) in SCALED_MM_TILES, f"{name} is not instantiated in the kernel"
