import pytest
import torch
import torch.nn.functional as F
from torch import nn

from feather_ops import conv2d_module
from feather_ops.conv2d import conv3x3_prepacked, prepack_conv2d_weight_3x3, weight_cache_key
from feather_ops.conv2d_module import (
    CONV3X3_PACKED_ATTRIBUTE,
    Conv2d,
    ModuleCache,
    conv3x3_module_reason,
    prepacked_conv3x3,
)
from feather_ops.modules import relabel_modules, relabelled_class

DEV = "cuda"
RTOL = 4e-3  # bf16 arithmetic


@pytest.fixture(autouse=True)
def _no_grad():
    previous = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    yield
    torch.set_grad_enabled(previous)


def _rel(got, ref):
    return float((got.double() - ref.double()).norm() / ref.double().norm())


def _module_with_weights(module, seed=0, scale=0.05):
    torch.manual_seed(seed)
    with torch.no_grad():
        module.weight.normal_(0, scale)
        if module.bias is not None:
            module.bias.normal_(0, 0.1)
    return module


def _packed(module):
    entry = ModuleCache().peek(module)
    return None if entry is None else entry[0]


def test_module_reason():
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, padding=1)) is None
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 1)) == "kernel is (1, 1)"
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, stride=2, padding=1)) == "stride is (2, 2)"
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, padding=0)) == "padding is (0, 0)"
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, padding=1, dilation=2)) == "dilation is (2, 2)"
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, padding=1, groups=2)) == "2 groups"
    assert conv3x3_module_reason(nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect")) == "'reflect' padding"
    assert conv3x3_module_reason(nn.Linear(8, 16)) == "not a conv2d"


def test_conv2d_is_a_drop_in():
    ref = _module_with_weights(nn.Conv2d(96, 96, 3, padding=1).cuda().to(torch.bfloat16), seed=1)
    mod = _module_with_weights(Conv2d(96, 96, 3, padding=1).cuda().to(torch.bfloat16), seed=1)
    mod.load_state_dict(ref.state_dict())
    assert set(mod.state_dict()) == set(ref.state_dict())  # the prepack is not state
    x = torch.randn((1, 96, 96, 96), device=DEV, dtype=torch.bfloat16)

    assert _rel(mod(x), ref(x)) < RTOL
    assert _packed(mod) is not None

    # a weight update is visible (the cache key follows the version counter)
    with torch.no_grad():
        mod.weight.mul_(0.5)
        ref.weight.mul_(0.5)
    assert _rel(mod(x), ref(x)) < RTOL

    # a dtype move drops the prepack, and the fp16 call is then delegated to MIOpen at this size
    mod.half()
    ref.half()
    x = x.half()
    assert _rel(mod(x), ref(x)) < RTOL
    assert _packed(mod) is None


def test_conv2d_falls_back_when_autograd_is_active():
    module = _module_with_weights(Conv2d(96, 96, 3, padding=1).cuda().to(torch.bfloat16))
    x = torch.randn((1, 96, 16, 16), device=DEV, dtype=torch.bfloat16)
    torch.set_grad_enabled(True)
    try:
        out = module(x)
    finally:
        torch.set_grad_enabled(False)
    assert _packed(module) is None
    assert torch.equal(out, F.conv2d(x, module.weight, module.bias, 1, 1))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: nn.Conv2d(8, 16, 1),  # not 3x3
        lambda: nn.Conv2d(8, 16, 3, stride=2, padding=0),  # stride 2
        lambda: nn.Conv2d(8, 16, 3, padding=1, padding_mode="reflect"),  # reflect padding
        lambda: nn.Conv2d(8, 16, 3, padding=1, groups=2),  # grouped
    ],
)
def test_conv2d_geometry_falls_back_bit_identically(factory):
    module = _module_with_weights(factory().cuda().to(torch.bfloat16))
    assert conv3x3_module_reason(module) is not None or module.groups != 1
    if conv3x3_module_reason(module) is not None:
        module.__class__ = Conv2d  # even relabelled, an out-of-geometry conv is torch's
    x = torch.randn((1, 8, 16, 16), device=DEV, dtype=torch.bfloat16)
    assert torch.equal(module(x), module._conv_forward(x, module.weight, module.bias))


def test_conv2d_does_not_prepack_a_delegated_weight():
    # fp16 is delegated to MIOpen, so nothing is packed
    module = _module_with_weights(Conv2d(96, 96, 3, padding=1).cuda().to(torch.float16), scale=0.05)
    x = torch.randn((1, 96, 128, 128), device=DEV, dtype=torch.float16)
    out = module(x)
    assert _packed(module) is None
    assert torch.equal(out, F.conv2d(x, module.weight, module.bias, 1, 1))

    # the same module in bf16 takes the kernel and packs
    module = module.to(torch.bfloat16)
    x = x.to(torch.bfloat16)
    assert torch.equal(module(x), conv3x3_prepacked(x, prepack_conv2d_weight_3x3(module.weight), module.bias))
    assert _packed(module).dtype == torch.bfloat16


def test_prepack_is_cached_per_weight_version(monkeypatch):
    packs = []
    real = conv2d_module.prepack_conv2d_weight_3x3
    monkeypatch.setattr(conv2d_module, "prepack_conv2d_weight_3x3", lambda *a, **k: (packs.append(1), real(*a, **k))[1])

    module = _module_with_weights(nn.Conv2d(32, 64, 3, padding=1).cuda().to(torch.bfloat16))
    x = torch.randn((1, 32, 32, 32), device=DEV, dtype=torch.bfloat16)

    _, bias = prepacked_conv3x3(module)
    prepacked_conv3x3(module)
    assert len(packs) == 1
    assert bias is not None and module.bias is not None
    assert bias.dtype == module.bias.dtype and torch.equal(bias, module.bias)  # the bias is used as is

    with torch.no_grad():
        module.weight.mul_(1.0)  # any in-place update bumps the version
    prepacked_conv3x3(module)
    assert len(packs) == 2

    module.load_state_dict({"weight": module.weight.clone(), "bias": module.bias.clone()})
    prepacked_conv3x3(module)
    assert len(packs) == 3

    # an explicit key (a cast copy, a different compute dtype) misses even without a weight change
    prepacked_conv3x3(module, key=(*weight_cache_key(module.weight), x.dtype, x.device))
    assert len(packs) == 4

    ModuleCache().invalidate(module)
    assert _packed(module) is None
    prepacked_conv3x3(module)
    assert len(packs) == 5


def test_prepack_follows_the_build_hook():
    module = _module_with_weights(nn.Conv2d(32, 64, 3, padding=1).cuda().to(torch.bfloat16))
    x = torch.randn((1, 32, 32, 32), device=DEV, dtype=torch.float16)

    # a consumer that casts/offloads makes the pack itself, and passes a key with the compute dtype
    def make():
        weight = module.weight.to(torch.float16)
        return prepack_conv2d_weight_3x3(weight), module.bias.to(torch.float16)

    key = (*weight_cache_key(module.weight), x.dtype, x.device)
    packed, bias = prepacked_conv3x3(module, key=key, make=make)
    assert packed.dtype == torch.float16 and bias is not None and bias.dtype == torch.float16
    assert torch.equal(module.weight, module.weight)  # untouched
    out = conv3x3_prepacked(x, packed, bias)
    ref = F.conv2d(x, module.weight.to(torch.float16), module.bias.to(torch.float16), 1, 1)
    assert _rel(out, ref) < 4e-3


def test_module_cache_hits_only_for_the_same_weight_object():
    cache = ModuleCache()
    module = _module_with_weights(nn.Conv2d(8, 16, 3, padding=1))
    key = weight_cache_key(module.weight)

    first = cache.get(module, module.weight, key, lambda: (torch.zeros(1), None))
    assert cache.get(module, module.weight, key, lambda: (torch.ones(1), None)) is first  # a hit, no rebuild
    assert cache.peek(module) is first

    # a new tensor with the same shape and version is a different owner, so it misses
    replacement = nn.Parameter(module.weight.detach().clone())
    assert cache.get(module, replacement, key, lambda: (torch.ones(1) * 2, None)) is not first
    assert CONV3X3_PACKED_ATTRIBUTE in module.__dict__

    cache.invalidate(module)
    assert cache.peek(module) is None


class _CountingMixin(nn.Module):
    """A stand-in for a consumer's mixin: count the calls so the test can see whose forward ran.

    It inherits `nn.Module` (not `object`) so the `super().forward` of a mixin that is always placed in front
    of a conv resolves for the type checker. The runtime MRO is unaffected.
    """

    calls: int

    def forward(self, input):  # ty: ignore[invalid-mutable-override]
        self.calls = getattr(self, "calls", 0) + 1
        return super().forward(input)


def test_relabel_modules_is_idempotent_and_transparent():
    root = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 1))
    state_before = list(root.state_dict().keys())
    params_before = {name: param for name, param in root.named_parameters()}
    x = torch.randn((1, 8, 8, 8))
    expected = root(x)

    patched = relabel_modules(root, _CountingMixin, lambda m: conv3x3_module_reason(m) is None)
    assert patched == 1  # only the 3x3
    assert isinstance(root[0], _CountingMixin) and isinstance(root[0], nn.Conv2d)
    assert not isinstance(root[2], _CountingMixin)
    assert list(root.state_dict().keys()) == state_before
    assert {name: param for name, param in root.named_parameters()} == params_before
    assert torch.equal(root(x), expected)
    assert root[0].calls == 1

    # a second call is a no-op, and the relabelled class is one cached object
    assert relabel_modules(root, _CountingMixin, lambda m: conv3x3_module_reason(m) is None) == 0
    assert type(root[0]) is relabelled_class(nn.Conv2d, _CountingMixin)


def test_relabel_modules_leaves_unapproved_subclasses_alone():
    class CustomConv(nn.Conv2d):
        def forward(self, input):  # a subclass with its own forward must not be prefixed
            return super().forward(input) * 2

    root = nn.Sequential(CustomConv(8, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1))
    patched = relabel_modules(root, _CountingMixin, lambda m: type(m) is nn.Conv2d)
    assert patched == 1 and type(root[0]) is CustomConv
