import pytest
import torch

from feather_ops.int8_convrot import (
    build_hadamard,
    convrot_int8_dgrad,
    convrot_int8_linear,
    convrot_int8_supported,
    dequantize_convrot_weight,
    int8_scaled_mm,
    prepack_int8_weight,
    quantize_convrot_activation,
    quantize_convrot_weight,
    quantize_rowwise_int8,
    rotate,
    select_int8_config,
    unpack_int8_weight,
)
from feather_ops.tiles import INT8_TILES

DEV = "cuda"
RTOL = 0.02  # bf16 output rounding of a scale multiply against an fp64 reference

# (M, N_w, K_w): the weight's output/input axes. The forward contracts K_w, the backward contracts N_w.
MODEL_CELLS = [
    (512, 2048, 1024),  # Anima cross-attn k/v
    (4096, 2048, 2048),  # Anima self q/k/v/out
    (4096, 8192, 2048),  # Anima mlp.0
    (4096, 2048, 8192),  # Anima mlp.2
    (4112, 1536, 6144),  # Krea 2 wk/wv
    (4112, 6144, 6144),  # Krea 2 wq/wo
    (4128, 11520, 3840),  # Z-Image
    (5376, 21504, 5376),  # H3
    (4112, 3072, 8192),  # MOSS-Image ffn_down
    (2048, 4304, 1152),  # Qwen3.5 text encoder, partial N
    (1024, 3456, 1152),  # Qwen3.5 text encoder, partial N
    (1, 36864, 6144),  # single-row prefix pass
    (16, 128, 128),
]


def _quantized(min_val: int, max_val: int, shape: tuple[int, ...], seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randint(min_val, max_val, shape, device=DEV, dtype=torch.int8, generator=gen)


def _scales(size: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    return torch.rand(size, device=DEV, dtype=torch.float32, generator=gen) * 0.02 + 1e-4


def _reference_fwd(a, wq, a_scale, w_scale, bias, out_dtype):
    acc = a.to(torch.float64) @ wq.to(torch.float64).t()
    acc = acc * a_scale.to(torch.float64)[:, None] * w_scale.to(torch.float64)[None, :]
    if bias is not None:
        acc = acc + bias.to(torch.float64)[None, :]
    return acc.to(out_dtype)


def _reference_dgrad(dy, wq, dy_scale, out_dtype):
    acc = dy.to(torch.float64) @ wq.to(torch.float64)
    return (acc * dy_scale.to(torch.float64)[:, None]).to(out_dtype)


def _max_rel(got, ref):
    return ((got.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-3)).max().item()


def test_prepack_round_trip_is_lossless():
    for n, k in [(4, 4), (96, 512), (2048, 8192), (4304, 1152)]:
        wq = _quantized(-127, 128, (n, k), seed=n + k)
        packed = prepack_int8_weight(wq)
        assert packed.shape == (k // 4, n // 4, 4, 4)
        assert packed.dtype == torch.int8
        # a single copy: the permutation does not change the number of bytes
        assert packed.numel() == wq.numel()
        assert torch.equal(unpack_int8_weight(packed), wq)


def test_prepack_requires_a_4x4_tile():
    # the layout is checked by the view (and the extent by the op), so this is a runtime layout error
    with pytest.raises(RuntimeError):
        prepack_int8_weight(torch.zeros(6, 8, dtype=torch.int8, device=DEV))


@pytest.mark.parametrize("out_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_bias", [False, True])
def test_forward_matches_dequantized_reference(out_dtype, has_bias):
    M, N, K = 512, 5888, 4096
    wq = _quantized(-127, 128, (N, K), seed=1)
    packed = prepack_int8_weight(wq)
    a = _quantized(-127, 128, (M, K), seed=2)
    a_scale = _scales(M, seed=3)
    w_scale = _scales(N, seed=4)
    bias = torch.randn(N, device=DEV, dtype=out_dtype) if has_bias else None
    got = int8_scaled_mm(a, packed, a_scale, w_scale, bias, out_dtype, False)
    ref = _reference_fwd(a, wq, a_scale, w_scale, bias, out_dtype)
    assert got.dtype == out_dtype
    assert _max_rel(got, ref) < RTOL


def test_backward_matches_dequantized_reference():
    M, N, K = 512, 5888, 4096
    wq = _quantized(-127, 128, (N, K), seed=5)
    packed = prepack_int8_weight(wq)
    dy = _quantized(-127, 128, (M, N), seed=6)
    dy_scale = _scales(M, seed=7)
    got = int8_scaled_mm(dy, packed, dy_scale, None, None, torch.bfloat16, True)
    ref = _reference_dgrad(dy, wq, dy_scale, torch.bfloat16)
    assert got.shape == (M, K)
    assert _max_rel(got, ref) < RTOL


def test_backward_rejects_a_weight_scale():
    packed = prepack_int8_weight(_quantized(-127, 128, (64, 64), seed=8))
    a = _quantized(-127, 128, (8, 64), seed=9)
    with pytest.raises(ValueError, match="fold"):
        int8_scaled_mm(a, packed, _scales(8, 10), _scales(64, 11), None, torch.bfloat16, True)


@pytest.mark.parametrize(("M", "N_w", "K_w"), MODEL_CELLS)
@pytest.mark.parametrize("transposed_b", [False, True])
def test_model_cells_against_reference(M, N_w, K_w, transposed_b):
    contraction, out = (N_w, K_w) if transposed_b else (K_w, N_w)
    wq = _quantized(-127, 128, (N_w, K_w), seed=0)
    packed = prepack_int8_weight(wq)
    a = _quantized(-127, 128, (M, contraction), seed=1)
    a_scale = _scales(M, seed=2)
    if transposed_b:
        got = int8_scaled_mm(a, packed, a_scale, None, None, torch.bfloat16, True)
        ref = _reference_dgrad(a, wq, a_scale, torch.bfloat16)
    else:
        w_scale = _scales(out, seed=3)
        got = int8_scaled_mm(a, packed, a_scale, w_scale, None, torch.bfloat16, False)
        ref = _reference_fwd(a, wq, a_scale, w_scale, None, torch.bfloat16)
    assert got.shape == (M, out)
    assert _max_rel(got, ref) < RTOL


@pytest.mark.parametrize("cfg", INT8_TILES)
@pytest.mark.parametrize("transposed_b", [False, True])
def test_every_table_config(cfg, transposed_b):
    warps_m, warps_n, unroll_k, repeat_m, repeat_n = cfg
    M = 16 * warps_m * repeat_m * 2
    N_w = 16 * warps_n * repeat_n * 2
    K_w = 16 * unroll_k * 3
    contraction, out = (N_w, K_w) if transposed_b else (K_w, N_w)
    wq = _quantized(-127, 128, (N_w, K_w), seed=12)
    packed = prepack_int8_weight(wq)
    a = _quantized(-127, 128, (M, contraction), seed=13)
    a_scale = _scales(M, seed=14)
    if transposed_b:
        got = int8_scaled_mm(a, packed, a_scale, None, None, torch.bfloat16, True, tile=cfg)
        ref = _reference_dgrad(a, wq, a_scale, torch.bfloat16)
    else:
        w_scale = _scales(out, seed=15)
        got = int8_scaled_mm(a, packed, a_scale, w_scale, None, torch.bfloat16, False, tile=cfg)
        ref = _reference_fwd(a, wq, a_scale, w_scale, None, torch.bfloat16)
    assert _max_rel(got, ref) < RTOL


@pytest.mark.parametrize(
    ("M", "N", "K", "expected_tile"),
    [
        # a row or two: 16 rows instead of 64 keeps four times the blocks in flight
        (1, 4096, 4096, (16, 64)),
        (16, 6144, 6144, (16, 64)),
        (1024, 1152, 1152, (256, 128)),
        (2048, 2048, 2048, (128, 128)),
        (4096, 8192, 2048, (256, 256)),
        (4112, 6144, 6144, (256, 256)),
        (5376, 21504, 5376, (256, 256)),
        # mid-width outputs with a real reduction axis take the 256x256 tile
        (5376, 5376, 7168, (256, 256)),  # H3 out_proj, +4.0% over 128x128
        (4128, 3840, 3840, (256, 256)),  # Z-Image, +7.4%
        (4128, 3840, 10240, (256, 256)),
        (4112, 3072, 3072, (256, 256)),  # MOSS-Image, +4.2%
        # ... except a power-of-two contraction at a small M (L2 aliasing with a 256-wide tile)
        (4112, 4096, 4096, (128, 128)),  # Qwen-Image / Klein at M=4112
        (2048, 4096, 4096, (128, 128)),
        # a short contraction keeps the 128x128 tile
        (2048, 4304, 1152, (128, 128)),
        (1024, 3456, 1152, (128, 128)),
        # an output below the mid-width class is unchanged
        (4096, 2048, 2048, (128, 128)),
        (4096, 2048, 8192, (128, 128)),
    ],
)
def test_shape_heuristic(M, N, K, expected_tile):
    warps_m, warps_n, _unroll, repeat_m, repeat_n = select_int8_config(M, N, K, False)
    assert (16 * warps_m * repeat_m, 16 * warps_n * repeat_n) == expected_tile


@pytest.mark.parametrize(
    "M, N, K, transposed_b, expected_tile",
    [
        # small M: one 64x64 launch beats a 128-row launch plus its 64-row tail launch (two single-wave grids).
        (192, 2560, 2560, False, (64, 64)),  # +1.6x over 128x128
        (192, 2560, 2560, True, (64, 64)),  # +1.36x
        (256, 4096, 4096, False, (64, 64)),
        (128, 8192, 5120, True, (64, 64)),
        (32, 25600, 5120, False, (64, 64)),  # 32-token prompt
        (1, 96768, 2688, False, (16, 64)),  # H3 adaln, group 64: +4% over 64x64
        (16, 6144, 6144, True, (16, 64)),  # +15-20% over 64x64 in the gradient direction
        # ... but a wide output pays for the 128x128 tile's larger N panel even at these row counts
        (192, 12288, 4096, False, (128, 128)),  # the 64x64 tile loses 1.34x
        (192, 12288, 4096, True, (64, 64)),  # the gradient's output is K = 4096, not 12288
        (128, 12288, 4096, False, (128, 128)),  # 64x64 loses 1.38x. 256x256 loses 2x
        (256, 12288, 4096, False, (128, 128)),
        # M = 512: four 128-row tiles, so the output width decides. Below 8192 the 64x64 tile wins (+4% on
        # 4096x4096, +29% on 2048x1024), which is what the earlier 256x256 rule got wrong.
        (512, 4096, 4096, False, (64, 64)),
        (512, 4096, 4096, True, (64, 64)),
        (512, 2048, 1024, True, (64, 64)),
        # ... a long reduction or a huge output puts the weight traffic back in charge
        (512, 12288, 4096, False, (128, 128)),
        (512, 248320, 4096, False, (256, 256)),
        (4112, 4096, 4096, True, (128, 128)),  # ... while a mid-size M still prefers 128x128 (+3.5%)
        (9232, 4096, 4096, True, (256, 256)),
    ],
)
def test_shape_heuristic_small_m(M, N, K, transposed_b, expected_tile):
    warps_m, warps_n, _unroll, repeat_m, repeat_n = select_int8_config(M, N, K, transposed_b)
    assert (16 * warps_m * repeat_m, 16 * warps_n * repeat_n) == expected_tile


def test_shape_heuristic_uses_the_output_width_of_the_call():
    # the backward produces K, so a wide K makes it a wide-output call even when N is narrow
    assert select_int8_config(4096, 2048, 8192, False)[0:2] == (2, 4)  # forward: output 2048
    assert select_int8_config(4096, 2048, 8192, True)[0:2] == (4, 4)  # backward: output 8192


def test_shape_heuristic_mid_width_output_in_both_directions():
    # H3 out_proj (5376, 7168): forward output 5376, gradient output 7168. Both take the 256x256 tile
    assert select_int8_config(5376, 5376, 7168, False)[0:2] == (4, 4)
    assert select_int8_config(5376, 5376, 7168, True)[0:2] == (4, 4)
    # the power-of-two exception is keyed on the contraction and M, not on the direction
    assert select_int8_config(4112, 4096, 4096, False)[0:2] == (2, 4)
    assert select_int8_config(4112, 4096, 4096, True)[0:2] == (2, 4)
    assert select_int8_config(9232, 4096, 4096, False)[0:2] == (4, 4)
    assert select_int8_config(9232, 4096, 4096, True)[0:2] == (4, 4)


# activation quantizer (ConvRot producer)


def _hadamard(size: int, dtype=torch.bfloat16) -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=dtype, device=DEV)
    h, n = h4, 4
    while n < size:
        h = torch.kron(h, h4)
        n *= 4
    return h / (size**0.5)


def _reference_quantize(x: torch.Tensor, rotate: bool, col_scale=None):
    """The recipe `library/convrot_int8_kernels.py` implements with a matmul plus a row-wise quantize."""
    if col_scale is not None:
        x = (x.float() * col_scale.to(torch.float32)).to(x.dtype)
    y = x
    if rotate:
        groups = x.shape[1] // 256
        y = torch.matmul(x.reshape(-1, groups, 256), _hadamard(256, x.dtype)).reshape(x.shape)
    scale = (y.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp(min=1e-30)
    q = (y / scale).round_().clamp_(-128.0, 127.0).to(torch.int8)
    return q, scale.reshape(-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rotate", [True, False])
@pytest.mark.parametrize("shape", [(1, 256), (4, 512), (512, 1024), (1024, 2048), (300, 4096), (16, 8192), (8, 16384)])
def test_quantize_matches_the_reference_recipe(dtype, rotate, shape):
    M, K = shape
    torch.manual_seed(0)
    x = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 2.0).to(dtype)
    q, scale = quantize_convrot_activation(x, rotate=rotate)
    ref_q, ref_scale = _reference_quantize(x, rotate)
    assert q.shape == (M, K) and q.dtype == torch.int8
    assert scale.shape == (M,) and scale.dtype == torch.float32
    # the rotation runs in a different summation order, so the codes agree up to values that sit on a .5
    # boundary: compare the dequantized matrices and bound how many codes may flip
    dq = q.float() * scale[:, None]
    ref_dq = ref_q.float() * ref_scale[:, None]
    assert ((dq - ref_dq).norm() / ref_dq.norm()).item() < 1e-3
    assert (q != ref_q).float().mean().item() < 1e-3
    assert ((scale - ref_scale).abs() / ref_scale.abs()).max().item() < 1e-2


# Every group-256 activation width a ConvRot step produces across the models and shapes of
# docs/int8_convrot_prepacked_fwd_bwd_plan.md. A step quantizes both the weight's input axis (forward) and its
# output axis (input gradient), so both the K and the N of each (N, K) family appear here. 16384 is the last row
# small enough to be staged in LDS next to a full block of waves. Above that the kernel streams it instead.
PLAN_PRODUCER_WIDTHS = [
    1024,  # Anima mlp
    2048,  # Anima self-attn / mlp
    3072,  # MOSS-Image
    3840,  # Z-Image
    4096,  # Qwen-Image 2.1 / Klein 9B / Qwen3.5-9B text encoder
    5120,  # Qwen3-VL-32B text encoder
    5376,  # H3
    6144,  # Krea 2
    7168,  # H3
    8192,  # Anima / MOSS-Image / Qwen3-VL-32B text encoder
    10240,  # Z-Image
    11520,  # Z-Image
    12288,  # Qwen-Image 2.1 / Klein 9B
    14336,  # H3
    16384,  # Krea 2 / MOSS-Image / Klein 9B (last staged row)
    21504,  # H3
    24576,  # Qwen-Image 2.1 / Klein 9B
    25600,  # Qwen3-VL-32B text encoder
    28672,  # H3
    36864,  # Klein 9B
    248320,  # Qwen3.5-9B text encoder vocab projection
]


@pytest.mark.parametrize("K", PLAN_PRODUCER_WIDTHS)
@pytest.mark.parametrize("rotate", [True, False])
def test_quantize_every_plan_width(K, rotate):
    """The producer covers every width the plan's models run, streamed or staged, with the reference codes."""
    M = 3
    torch.manual_seed(K)
    x = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 2.0).to(torch.bfloat16)
    q, scale = quantize_convrot_activation(x, rotate=rotate)
    ref_q, ref_scale = _reference_quantize(x, rotate)
    dq = q.float() * scale[:, None]
    ref_dq = ref_q.float() * ref_scale[:, None]
    assert ((dq - ref_dq).norm() / ref_dq.norm()).item() < 1e-3
    assert (q != ref_q).float().mean().item() < 1e-3
    assert ((scale - ref_scale).abs() / ref_scale.abs()).max().item() < 1e-2


@pytest.mark.parametrize("K", [16384, 16640, 32768, 248320])  # 16384 is the staging threshold
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantize_wide_rows_fold_the_column_scale(K, dtype):
    """The input gradient's producer (no rotation, weight scale folded in) on the wide rows."""
    M = 5
    torch.manual_seed(K)
    x = (torch.randn(M, K, device=DEV, dtype=torch.float32) * 2.0).to(dtype)
    col_scale = (torch.rand(K, device=DEV, dtype=torch.float32) * 0.5 + 0.5).to(torch.float32)
    q, scale = quantize_convrot_activation(x, rotate=False, col_scale=col_scale)
    ref_q, ref_scale = _reference_quantize(x, False, col_scale)
    dq = q.float() * scale[:, None]
    ref_dq = ref_q.float() * ref_scale[:, None]
    assert ((dq - ref_dq).norm() / ref_dq.norm()).item() < 1e-3
    assert (q != ref_q).float().mean().item() < 1e-3


def test_quantize_folds_the_column_scale():
    """The gradient path folds the weight's per-output-channel scale into the quantizer."""
    M, K = 512, 2048
    torch.manual_seed(1)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    col_scale = torch.rand(K, device=DEV, dtype=torch.float32) * 0.5 + 0.5
    q, scale = quantize_convrot_activation(x, rotate=False, col_scale=col_scale)
    ref_q, ref_scale = _reference_quantize(x, False, col_scale)
    dq = q.float() * scale[:, None]
    ref_dq = ref_q.float() * ref_scale[:, None]
    assert ((dq - ref_dq).norm() / ref_dq.norm()).item() < 1e-3
    assert (q != ref_q).float().mean().item() < 1e-3


def test_quantize_rejects_a_non_multiple_of_the_group_size():
    x = torch.zeros(4, 384, device=DEV, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="multiple of the ConvRot group size"):
        quantize_convrot_activation(x)


def test_quantize_output_feeds_the_gemm():
    """End to end: quantize an activation, dequantize the weight, compare against the float product."""
    M, N, K = 512, 2048, 2048
    torch.manual_seed(2)
    wq = _quantized(-127, 128, (N, K), seed=3)
    w_scale = _scales(N, seed=4)
    packed = prepack_int8_weight(wq)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 0.1
    q, scale = quantize_convrot_activation(x, rotate=True)
    got = int8_scaled_mm(q, packed, scale, w_scale, None, torch.bfloat16, False)
    # W = W_rot @ R^T, i.e. the stored weight rotated back per group
    h = _hadamard(256, torch.float32)
    w_deq = torch.matmul((wq.float() * w_scale[:, None]).reshape(N, K // 256, 256), h).reshape(N, K)
    ref = x.float() @ w_deq.t()
    assert ((got.float() - ref).norm() / ref.norm()).item() < 2e-2


def test_compile_is_supported():
    """The schedule heuristic lives behind an opaque op, so a dynamic graph is one node and matches eager."""
    torch.manual_seed(0)
    m, n_w, k_w = 4112, 1152, 1152
    packed = prepack_int8_weight(torch.randint(-127, 128, (n_w, k_w), device=DEV, dtype=torch.int8))
    a = torch.randint(-127, 128, (m, k_w), device=DEV, dtype=torch.int8)
    a_scale = torch.rand(m, device=DEV, dtype=torch.float32) * 0.02
    w_scale = torch.rand(n_w, device=DEV, dtype=torch.float32) * 0.02

    def run(a, a_scale):
        return int8_scaled_mm(a, packed, a_scale, w_scale, None, torch.bfloat16, False)

    eager = run(a, a_scale)
    compiled = torch.compile(run, fullgraph=True, dynamic=True)(a, a_scale)
    assert torch.equal(compiled, eager)


def _reference_hadamard(size: int) -> torch.Tensor:
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h, n = h4, 4
    while n < size:
        h = torch.kron(h, h4)
        n *= 4
    return h / (size**0.5)


def _reference_quantize_weight(weight, rotate_dtype, clamp_min):
    """The two conventions that exist in the wild, written out: rotate in `rotate_dtype`, clamp at `clamp_min`."""
    hadamard = _reference_hadamard(256).to(weight.device)
    out_f, in_f = weight.shape
    grouped = weight.to(rotate_dtype).reshape(out_f, in_f // 256, 256)
    rotated = torch.matmul(grouped, hadamard.T.to(dtype=rotate_dtype)).reshape(out_f, in_f)
    scale = (rotated.abs().amax(dim=1).float() / 127.0).clamp(min=1e-30)
    q = torch.clamp(torch.round(rotated.float() / scale[:, None]), clamp_min, 127.0).to(torch.int8)
    return q, scale


def test_hadamard_matches_the_reference_construction():
    for size in (4, 16, 64, 256):
        got = build_hadamard(size)
        assert torch.equal(got, _reference_hadamard(size))
        assert torch.equal(got, got.T)  # symmetric, so it is its own inverse
        assert torch.equal(got @ got, torch.eye(size))  # orthogonal
    # cached, and a different dtype/device is a different matrix
    assert build_hadamard(256) is build_hadamard(256)
    assert build_hadamard(256, dtype=torch.float64).dtype == torch.float64
    with pytest.raises(ValueError, match="power of four"):
        build_hadamard(12)


@pytest.mark.parametrize("shape", [(4, 512), (64, 1024), (3, 256)])
def test_rotate_is_its_own_inverse(shape):
    x = torch.randn(shape, device=DEV, dtype=torch.float32)
    hadamard = build_hadamard(256, device=DEV)
    back = rotate(rotate(x, hadamard), hadamard)
    assert torch.allclose(back, x, atol=1e-5)
    with pytest.raises(ValueError, match="group size"):
        rotate(torch.randn(3, 300, device=DEV), hadamard)


def test_quantize_rowwise_int8_codes_and_clamp():
    x = torch.randn(8, 256, device=DEV, dtype=torch.float32)
    q, scale = quantize_rowwise_int8(x)
    ref_scale = (x.abs().amax(dim=1) / 127.0).clamp(min=1e-30)
    ref_q = torch.clamp(torch.round(x / ref_scale[:, None]), -127.0, 127.0).to(torch.int8)
    assert torch.equal(q, ref_q)
    assert torch.equal(scale, ref_scale)

    # A scale derived from the row's absmax already stays inside [-127, 127], so the clamp is only a guard:
    # no value, dtype or rounding can reach -128, which is why the -128 floor some weight recipes use is
    # unobservable (and why the clamp is not an argument).
    torch.manual_seed(0)
    adversarial = [
        -torch.ones(4, 256, device=DEV),  # the absmax is the most negative value
        torch.randn(64, 512, device=DEV) * 1e-30,
        torch.randn(64, 512, device=DEV) * 1e30,
        torch.randn(64, 512, device=DEV).clamp(-1e-4, 1.0),
    ]
    for dtype in (torch.bfloat16, torch.float16):
        adversarial += [t.to(dtype) for t in adversarial[:2]]
    for tensor in adversarial:
        codes = quantize_rowwise_int8(tensor)[0]
        assert codes.min() >= -127 and codes.max() <= 127

    # a stored scale can be out of range, and there the bound is the absmax's code, not -128
    x = torch.tensor([[-1.0, 0.5]], device=DEV)
    stored_scale = torch.tensor([1.0 / 130.0], device=DEV)  # -1 / scale = -130
    assert quantize_rowwise_int8(x, stored_scale)[0].min() == -127


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(64, 256), (64, 512), (128, 1024), (36, 2048), (1152, 1024)])
@pytest.mark.parametrize("rotate_dtype", [torch.float32, None])
def test_quantize_convrot_weight_matches_the_reference(dtype, shape, rotate_dtype):
    """The rotation dtype is the degree of freedom: fp32 (sd-scripts) or the weight's own dtype (comfy-quants)."""
    torch.manual_seed(0)
    weight = torch.randn(shape, device=DEV, dtype=torch.float32).to(dtype)
    q, scale = quantize_convrot_weight(weight, rotate_dtype=rotate_dtype)
    ref_q, ref_scale = _reference_quantize_weight(weight, rotate_dtype or dtype, clamp_min=-127.0)
    assert q.dtype == torch.int8 and q.shape == weight.shape
    assert torch.equal(q, ref_q)
    assert torch.equal(scale, ref_scale)
    # the -128 floor one reference implementation uses cannot change a single code on a derived scale
    assert torch.equal(q, _reference_quantize_weight(weight, rotate_dtype or dtype, clamp_min=-128.0)[0])


def test_dequantize_convrot_weight_round_trips_the_rotation():
    torch.manual_seed(1)
    weight = (torch.randn(64, 512, device=DEV, dtype=torch.float32) * 0.05).to(torch.bfloat16)
    q, scale = quantize_convrot_weight(weight, rotate_dtype=torch.float32)
    back = dequantize_convrot_weight(q, scale)
    relative = (back - weight.float()).norm() / weight.float().norm()
    # one int8 rounding per row: worst case 1/(2*127) of the row's absmax, scaled by absmax/rms
    assert relative < 1e-2
    assert back.dtype == torch.float32


def test_convrot_int8_supported():
    assert convrot_int8_supported((256, 1024)) == (True, "")
    assert convrot_int8_supported((256, 1024), backward=False) == (True, "")
    ok, reason = convrot_int8_supported((4, 1024))
    assert not ok and "16" in reason  # the input gradient contracts this axis
    assert convrot_int8_supported((4, 1024), backward=False)[0]
    ok, reason = convrot_int8_supported((256, 1024), groupsize=64)
    assert not ok and "group size" in reason
    assert not convrot_int8_supported((256, 1000))[0]  # K must be a multiple of 256
    assert not convrot_int8_supported((254, 1024))[0]  # N must be a multiple of 4


@pytest.mark.parametrize("shape", [(4, 256, 512), (32, 256, 1024)])
@pytest.mark.parametrize("weight_packed", [True, False])
def test_convrot_int8_linear_matches_the_composition(shape, weight_packed):
    M, n_out, k_in = shape
    w = _quantized(-127, 128, (n_out, k_in), seed=2)
    w_scale = _scales(n_out, seed=4)
    bias = torch.randn(n_out, device=DEV, dtype=torch.bfloat16) * 0.1

    x = torch.randn(M, k_in, device=DEV, dtype=torch.bfloat16)
    packed = prepack_int8_weight(w) if weight_packed else w
    got = convrot_int8_linear(x, packed, w_scale, bias, torch.bfloat16, weight_packed=weight_packed)
    q, s = quantize_convrot_activation(x, rotate=True)
    ref = int8_scaled_mm(q, prepack_int8_weight(w), s, w_scale, bias, torch.bfloat16, False)
    assert torch.equal(got, ref)


@pytest.mark.parametrize("weight_packed", [True, False])
def test_convrot_int8_dgrad_matches_the_composition(weight_packed):
    M, n_out, k_in = 32, 512, 1024
    w = _quantized(-127, 128, (n_out, k_in), seed=5)
    w_scale = _scales(n_out, seed=6)
    g = torch.randn(M, n_out, device=DEV, dtype=torch.bfloat16)

    packed = prepack_int8_weight(w) if weight_packed else w
    got = convrot_int8_dgrad(g, packed, w_scale, torch.bfloat16, weight_packed=weight_packed)
    q, s = quantize_convrot_activation(g, rotate=False, col_scale=w_scale)
    ref = int8_scaled_mm(q, prepack_int8_weight(w), s, None, None, torch.bfloat16, True)
    ref = rotate(ref, build_hadamard(256, device=DEV, dtype=torch.bfloat16))
    assert got.shape == (M, k_in)
    assert torch.equal(got, ref)


def test_convrot_int8_linear_leaves_the_shape_checks_to_the_op():
    x = torch.randn(4, 1000, device=DEV, dtype=torch.bfloat16)
    w = torch.zeros(256, 1000, device=DEV, dtype=torch.int8)
    with pytest.raises(RuntimeError, match="group size"):
        convrot_int8_linear(x, w, torch.ones(256, device=DEV))
    # a mismatched activation is caught here, because `n_out` comes from the weight's extent
    with pytest.raises(ValueError, match="features"):
        convrot_int8_linear(x, torch.zeros(256, 512, device=DEV, dtype=torch.int8), torch.ones(256, device=DEV))
