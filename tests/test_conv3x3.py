import pytest
import torch
import torch.nn.functional as F

from feather_ops.conv2d import Conv2d, conv2d, conv3x3_prepacked, prepack_conv2d_weight_3x3, select_backend, select_conv3x3_config
from feather_ops.tiles import CONV_TILES, conv_block_sizes

DTYPES = [torch.bfloat16, torch.float16]


@pytest.fixture(autouse=True)
def _no_grad():
    """The kernel is inference-only. Keep every test in eval mode and restore the caller's state."""
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    yield
    torch.set_grad_enabled(previous)


# (C_in, C_out, H, W): covers supported tiles, M/K tails, C_in not a multiple of 16, odd widths.
SHAPES = [
    (16, 16, 32, 32),
    (32, 32, 64, 64),
    (96, 96, 40, 64),
    (128, 128, 64, 64),
    (144, 144, 33, 96),
    (384, 384, 65, 128),
    (384, 96, 32, 192),
    (96, 384, 32, 256),
    (128, 128, 17, 512),
    (8, 16, 48, 64),
    (56, 512, 32, 64),
    (24, 96, 32, 96),
    (1152, 1152, 16, 32),
    (64, 48, 9, 32),
    # two output rows per block: 128-wide images, and C_out that needs the 64-wide M tile
    (384, 384, 40, 128),
    (192, 192, 33, 256),
    (576, 576, 17, 256),
    (384, 384, 1, 128),
    (384, 384, 3, 128),
]


def _tile(block_m: int, block_n: int, rows: int) -> tuple[int, int, int, int, int]:
    """The instantiation with this (block_m, block_n, rows) box."""
    return next(tile for tile in CONV_TILES if conv_block_sizes(tile) == (block_m, block_n, rows))


def _reference(x, weight, bias):
    return F.conv2d(x.double(), weight.double(), bias.double() if bias is not None else None, 1, 1)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(("c_in", "c_out", "h", "w"), SHAPES)
@pytest.mark.parametrize("with_bias", [True, False])
def test_conv3x3_matches_reference(dtype, c_in, c_out, h, w, with_bias):
    torch.manual_seed(c_in * 131 + c_out)
    x = torch.randn((2, c_in, h, w), device="cuda", dtype=torch.float32).to(dtype)
    weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
    bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(dtype) if with_bias else None

    tile = select_conv3x3_config(c_out, w)
    assert tile is not None, f"the shape matrix is supported, but C_out={c_out} W={w} has no tile"
    packed = prepack_conv2d_weight_3x3(weight)
    out = conv3x3_prepacked(x, packed, bias, tile=tile)
    ref = _reference(x, weight, bias)
    assert torch.isfinite(out).all()
    rel = float((out.double() - ref).norm() / ref.norm())
    tol = 4e-3 if dtype == torch.bfloat16 else 6e-4
    assert rel < tol, f"rel={rel} for {dtype} {c_in}->{c_out} {h}x{w}"


# C_out that no tile divides (40 = 8 * 5) is outside the kernel's contract: the dispatch must say so and
# `conv2d` must still be a drop-in for `F.conv2d` through the documented fallback.
@pytest.mark.parametrize(("c_in", "c_out", "h", "w"), [(40, 40, 33, 48), (24, 40, 16, 32)])
def test_unsupported_c_out_is_rejected_and_falls_back(c_in, c_out, h, w):
    assert select_conv3x3_config(c_out, w) is None
    # bf16 has no MIOpen Winograd, so it reaches torch. fp16 is delegated while the input is small.
    assert select_backend(torch.bfloat16, c_in, c_out, h, w) == "torch"
    assert select_backend(torch.float16, c_in, c_out, h, w) == "miopen"
    for dtype in DTYPES:
        x = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.float32).to(dtype)
        weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
        got = conv2d(x, weight, None, 1, 1)
        ref = _reference(x, weight, None)
        assert float((got.double() - ref).norm() / ref.norm()) < 4e-3


def test_prepack_roundtrip_is_a_permutation():
    for c_in, c_out in [(16, 8), (32, 33), (144, 96), (24, 5), (21, 5)]:
        weight = torch.randn((c_out, c_in, 3, 3))
        packed = prepack_conv2d_weight_3x3(weight)
        groups = (c_in + 15) // 16
        assert packed.shape == (groups * 18, c_out, 8)
        k = packed.permute(0, 2, 1).reshape(groups * 9 * 16, c_out)
        back = k.reshape(groups, 9, 16, c_out).permute(3, 0, 2, 1).reshape(c_out, groups * 16, 9)
        assert torch.equal(back[:, :c_in], weight.reshape(c_out, c_in, 9))
        if groups * 16 != c_in:
            assert torch.equal(back[:, c_in:], torch.zeros_like(back[:, c_in:]))


def test_functional_api_matches_torch_and_falls_back():
    torch.manual_seed(0)
    for dtype in DTYPES:
        x = torch.randn((1, 96, 64, 64), device="cuda", dtype=torch.float32).to(dtype)
        weight = (torch.randn((96, 96, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(dtype)
        bias = torch.zeros((96,), device="cuda", dtype=dtype)
        got = conv2d(x, weight, bias, 1, 1)
        ref = F.conv2d(x, weight, bias, 1, 1)
        assert float((got.double() - ref.double()).norm() / ref.double().norm()) < 4e-3
    # stride 2 falls back to torch.
    x = torch.randn((1, 96, 33, 33), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((96, 96, 3, 3), device="cuda", dtype=torch.bfloat16)
    assert torch.equal(conv2d(x, weight, None, 2, 0), F.conv2d(x, weight, None, 2, 0))


def test_module_is_a_drop_in():
    torch.manual_seed(1)
    ref_mod = torch.nn.Conv2d(96, 96, 3, padding=1, bias=True).cuda().to(torch.bfloat16)
    mod = Conv2d(96, 96, 3, padding=1, bias=True).cuda().to(torch.bfloat16)
    mod.load_state_dict(ref_mod.state_dict())
    assert set(mod.state_dict()) == set(ref_mod.state_dict())
    x = torch.randn((1, 96, 96, 96), device="cuda", dtype=torch.bfloat16)
    ref = ref_mod(x).detach()
    out = mod(x)
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3
    assert mod._packed_weight.numel() > 0
    # a weight update invalidates the prepack
    with torch.no_grad():
        mod.weight.mul_(0.5)
        ref_mod.weight.mul_(0.5)
    out = mod(x)
    ref = ref_mod(x).detach()
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3
    # a dtype move invalidates it too
    mod.half()
    x = x.half()
    ref_mod.half()
    out = mod(x)
    ref = ref_mod(x).detach()
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_module_falls_back_when_autograd_is_active():
    module = Conv2d(96, 96, 3, padding=1, bias=True).cuda().to(torch.bfloat16)
    x = torch.randn((1, 96, 32, 32), device="cuda", dtype=torch.bfloat16)
    torch.set_grad_enabled(True)
    out = module(x)
    torch.set_grad_enabled(False)
    assert module._packed_weight.numel() == 0
    ref = F.conv2d(x, module.weight, module.bias, 1, 1)
    assert torch.equal(out, ref)


def test_odd_shapes_and_batch():
    torch.manual_seed(2)
    for shape in [(1, 48, 1, 1), (3, 32, 5, 7), (2, 16, 3, 33), (1, 128, 129, 33), (2, 128, 65, 128)]:
        b, c, h, w = shape
        x = torch.randn((b, c, h, w), device="cuda", dtype=torch.bfloat16)
        weight = (torch.randn((c, c, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        out = conv2d(x, weight, None, 1, 1)
        ref = F.conv2d(x, weight, None, 1, 1)
        assert out.shape == ref.shape
        assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3


def test_two_row_tiles_are_selected_where_measured():
    """rows=2 is a heuristic over (C_out, width). Pin it so a tile-table edit cannot move it silently."""
    two_rows = {
        (1152, 128): (128, 128, 2),
        (384, 128): (128, 128, 2),
        (512, 128): (128, 128, 2),
        (192, 256): (64, 256, 2),
        (576, 256): (64, 256, 2),
        (192, 512): (64, 256, 2),
    }
    one_row = {
        (128, 256): (128, 256, 1),
        (256, 256): (128, 256, 1),
        (512, 256): (128, 256, 1),
        (96, 256): (96, 256, 1),
        (144, 256): (48, 256, 1),
        (288, 256): (96, 256, 1),
        (96, 128): (96, 128, 1),
        (192, 128): (96, 128, 1),
        (144, 128): (48, 128, 1),
        (96, 64): (96, 64, 1),
    }
    for (c_out, width), expected in {**two_rows, **one_row}.items():
        tile = select_conv3x3_config(c_out, width)
        assert tile is not None and conv_block_sizes(tile) == expected, (c_out, width)


def test_two_rows_matches_one_row_exactly():
    """Doubling the rows must not change any pixel's reduction order, so the outputs are identical."""
    torch.manual_seed(4)
    for c_out, c_in, h, w, box in [(384, 384, 40, 128, (128, 128)), (192, 192, 33, 256, (64, 256))]:
        x = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.bfloat16)
        weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
        bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16)
        packed = prepack_conv2d_weight_3x3(weight)
        one = conv3x3_prepacked(x, packed, bias, tile=_tile(*box, 1))
        two = conv3x3_prepacked(x, packed, bias, tile=_tile(*box, 2))
        assert torch.equal(one, two), (c_out, h, w)


def _miopen_conv2d(x, weight, bias):
    """`F.conv2d` with MIOpen forced on, which is what the delegated path must produce."""
    enabled = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = True
    try:
        return F.conv2d(x, weight, bias, 1, 1)
    finally:
        torch.backends.cudnn.enabled = enabled


def test_select_backend():
    """fp16 3x3 s1 is delegated to MIOpen, bf16 is not, and the tile table decides the rest."""
    # fp16 inside the delegation footprint window
    assert select_backend(torch.float16, 512, 512, 256, 256) == "miopen"
    assert select_backend(torch.float16, 96, 96, 128, 128) == "miopen"
    # ... even where the kernel has no tile at all, since MIOpen does not care
    assert select_backend(torch.float16, 512, 512, 16, 16) == "miopen"
    # bf16 never is: MIOpen has no bf16 Winograd
    assert select_backend(torch.bfloat16, 512, 512, 256, 256) == "kernel"
    # beyond the window MIOpen's Winograd loses, so fp16 goes back to the kernel
    assert select_backend(torch.float16, 128, 128, 2048, 2048) == "kernel"
    # no tile and not delegated -> plain torch
    assert select_backend(torch.bfloat16, 512, 512, 16, 16) == "torch"
    assert select_backend(torch.float32, 512, 512, 256, 256) == "torch"


def test_fp16_conv2d_is_delegated_to_miopen():
    torch.manual_seed(11)
    x = torch.randn((1, 96, 128, 128), device="cuda", dtype=torch.float16)
    weight = (torch.randn((96, 96, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(torch.float16)
    bias = (torch.randn((96,), device="cuda", dtype=torch.float32) * 0.1).to(torch.float16)
    out = conv2d(x, weight, bias, 1, 1)
    assert torch.equal(out, _miopen_conv2d(x, weight, bias)), "the delegated path is F.conv2d + MIOpen"
    # and it really is not this kernel: fp16 Winograd vs direct differ at the bf16-arithmetic level
    kernel = conv3x3_prepacked(x, prepack_conv2d_weight_3x3(weight), bias)
    assert not torch.equal(out, kernel)


def test_module_does_not_prepack_a_delegated_weight():
    torch.manual_seed(12)
    module = Conv2d(96, 96, 3, padding=1, bias=True).cuda().to(torch.float16)
    with torch.no_grad():
        module.weight.normal_(0, 0.05)
        assert module.bias is not None
        module.bias.normal_(0, 0.1)
    x = torch.randn((1, 96, 128, 128), device="cuda", dtype=torch.float16)
    assert torch.equal(module(x), _miopen_conv2d(x, module.weight, module.bias))
    assert module._packed_weight.numel() == 0, "a delegated weight must not be prepacked"

    # the same module in bf16 keeps the kernel path and does prepack
    module = module.to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    packed = prepack_conv2d_weight_3x3(module.weight)
    assert torch.equal(module(x), conv3x3_prepacked(x, packed, module.bias))
    assert module._packed_weight.numel() > 0


def test_dispatch_matrix_matches_torch():
    """Every (C_out, W) the selector can route must produce the right numbers, not just a shape."""
    torch.manual_seed(5)
    c_in, h = 32, 7
    for c_out in (16, 32, 48, 64, 96, 128, 144, 192, 256, 288, 384, 512, 576, 1152):
        for w in (32, 64, 96, 128, 192, 256):
            x = torch.randn((1, c_in, h, w), device="cuda", dtype=torch.bfloat16)
            weight = (torch.randn((c_out, c_in, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
            bias = (torch.randn((c_out,), device="cuda", dtype=torch.float32) * 0.1).to(torch.bfloat16)
            out = conv2d(x, weight, bias, 1, 1)
            ref = F.conv2d(x, weight, bias, 1, 1)
            rel = float((out.double() - ref.double()).norm() / ref.double().norm())
            assert out.shape == ref.shape and rel < 4e-3, (c_out, w, rel)


def test_supported_tiles_are_instantiated():
    """Every (block_m, block_n) the selector can return must exist in the kernel."""
    assert len(CONV_TILES) == 38  # 36 one-row tiles and the two two-row ones
    for c_out in (16, 32, 48, 64, 96, 128, 144, 192, 256, 288, 384, 512, 576, 768, 1152):
        for w in (32, 64, 96, 128, 192, 256, 384, 512, 1024):
            tile = select_conv3x3_config(c_out, w)
            if tile is None:
                continue
            block_m, block_n, rows = conv_block_sizes(tile)
            assert tile in CONV_TILES
            assert c_out % block_m == 0 and w % block_n == 0
            # both row counts must exist for the tiles the selector can ask for
            x = torch.zeros((1, 16, max(w, 32), max(w, 32)), device="cuda", dtype=torch.bfloat16)
            packed = torch.zeros((18, c_out, 8), device="cuda", dtype=torch.bfloat16)
            for rows_wanted in {1, rows}:
                out = conv3x3_prepacked(x, packed, None, tile=_tile(block_m, block_n, rows_wanted))
                assert out.shape == (1, c_out, max(w, 32), max(w, 32))


def test_compile_is_supported():
    torch.manual_seed(3)
    x = torch.randn((1, 96, 64, 64), device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn((96, 96, 3, 3), device="cuda", dtype=torch.float32) * 0.05).to(torch.bfloat16)
    packed = prepack_conv2d_weight_3x3(weight)
    tile = select_conv3x3_config(96, 64)

    assert tile is not None

    def run(inp, wk):
        return conv3x3_prepacked(inp, wk, None, tile=tile)

    compiled = torch.compile(run, fullgraph=True)
    out = compiled(x, packed)
    ref = F.conv2d(x, weight, None, 1, 1)
    assert float((out.double() - ref.double()).norm() / ref.double().norm()) < 4e-3
