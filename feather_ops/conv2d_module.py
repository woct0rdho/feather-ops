"""Running the 3x3 conv kernel inside modules that already exist.

A consumer that patches an existing model tree (ComfyUI's VAE loader, sd-scripts' trainer) writes its
own mixin, because the parts that differ are framework facts: how a cast or offloaded weight is
obtained, whether a patch is installed on the weight, memory hooks, logging. It uses
`conv3x3_module_reason` for the static check, `prepacked_conv3x3` for the prepack and
`feather_ops.modules.relabel_modules` to put its mixin in front of `forward`.
"""

from collections.abc import Callable

import torch
from torch import nn

from .conv2d import (
    conv2d_miopen,
    conv3x3_eligibility,
    conv3x3_prepacked,
    prepack_conv2d_weight_3x3,
    select_backend,
    weight_cache_key,
)

CONV3X3_PACKED_ATTRIBUTE = "_feather_conv3x3_packed"

# What a module caches: the prepacked weight and a contiguous bias in the weight's dtype (None without one).
Conv3x3Prepack = tuple[torch.Tensor, torch.Tensor | None]


def _padding(module: nn.Conv2d) -> tuple[int, ...]:
    """`module.padding` as integers. torch also allows a str (`"same"`), which is never the kernel's case."""
    padding = module.padding
    return () if isinstance(padding, str) else tuple(int(p) for p in padding)


def conv3x3_module_reason(module: nn.Module, padding: tuple[int, ...] | None = None) -> str | None:
    """Why this module's convolutions can never use the kernel, or None if the geometry is right.

    This is the part of the decision that does not depend on the call: kernel size, stride, padding,
    dilation, groups, padding mode. `conv3x3_eligibility` decides the rest (dtype, layout, autograd).

    `padding` overrides `module.padding` for classes that zero it and pad the input in their own forward
    (`ChunkedConv2d` and friends), where the effective padding is a separate attribute.
    """
    if not isinstance(module, nn.Conv2d):
        return "not a conv2d"
    effective_padding = tuple(int(p) for p in padding) if padding is not None else _padding(module)
    if tuple(module.kernel_size) != (3, 3):
        return f"kernel is {tuple(module.kernel_size)}"
    if tuple(module.stride) != (1, 1):
        return f"stride is {tuple(module.stride)}"
    if effective_padding != (1, 1):
        return f"padding is {effective_padding}"
    if tuple(module.dilation) != (1, 1):
        return f"dilation is {tuple(module.dilation)}"
    if module.groups != 1:
        return f"{module.groups} groups"
    if module.padding_mode != "zeros":
        return f"{module.padding_mode!r} padding"
    return None


class ModuleCache:
    """The conv prepack of a module, attached to the module and rebuilt when the key changes.

    The entry holds the weight object itself and only hits for that same object with that same key,
    because a freed tensor's address can be reused by a new tensor that also starts at version 0. Keys
    must not contain tensors (tuple comparison would go elementwise): `weight_cache_key` plus the
    compute dtype/device is what callers pass.

    The value lives in `module.__dict__`, not in a registered buffer: it is a build product, so
    `state_dict` must not see it, and a dtype/device move changes the key anyway. Consumers can also
    call `invalidate` from `_apply` to free it early.
    """

    def __init__(self, attribute: str = CONV3X3_PACKED_ATTRIBUTE) -> None:
        self.attribute = attribute

    def get(self, module: nn.Module, weight: torch.Tensor, key: tuple, build: Callable[[], Conv3x3Prepack]) -> Conv3x3Prepack:
        """The cached prepack for `(weight, key)`, built by `build()` on a miss."""
        entry = module.__dict__.get(self.attribute)
        if entry is not None and entry[0] is weight and entry[1] == key:
            return entry[2]
        value = build()
        module.__dict__[self.attribute] = (weight, key, value)
        return value

    def peek(self, module: nn.Module) -> Conv3x3Prepack | None:
        """The cached prepack, or None when nothing has been packed for this module yet."""
        entry = module.__dict__.get(self.attribute)
        return None if entry is None else entry[2]

    def invalidate(self, module: nn.Module) -> None:
        module.__dict__.pop(self.attribute, None)


_CACHE = ModuleCache()


def _pack(weight: torch.Tensor, bias: torch.Tensor | None) -> Conv3x3Prepack:
    packed_bias = None if bias is None else bias.detach().to(device=weight.device, dtype=weight.dtype).contiguous()
    return prepack_conv2d_weight_3x3(weight.detach()), packed_bias


def prepacked_conv3x3(module: nn.Conv2d, key: tuple | None = None, make: Callable[[], Conv3x3Prepack] | None = None) -> Conv3x3Prepack:
    """`(prepacked weight, contiguous bias)` for `module`, cached on the module's weight version.

    `make()` returns `(prepacked weight, bias)` and is for a consumer whose kernel weight is not the
    module's own parameter - a runtime that casts or offloads it per call. Such a caller passes a `key`
    that includes the compute dtype and device, because the module's weight does not change when only
    the cast copy does. The default packs the module's weight and copies its bias.
    """
    if key is None:
        key = weight_cache_key(module.weight)
    if make is None:
        make = lambda: _pack(module.weight, module.bias)
    return _CACHE.get(module, module.weight, key, make)


class Conv2d(nn.Conv2d):
    """`nn.Conv2d` that runs the feather kernel when the call is in scope, torch's path otherwise.

    A drop-in: `state_dict` is unchanged (the prepack is a build product, not a buffer) and every
    fallback is `super().forward()`, i.e. exactly what the module did before.
    """

    def _apply(self, fn, recurse: bool = True):
        # a dtype/device move changes the key, but dropping the entry now frees the old copy instead of
        # keeping it until the next kernel-path call (which a later delegation may never make)
        result = super()._apply(fn, recurse)
        _CACHE.invalidate(self)
        return result

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if conv3x3_module_reason(self) is not None or conv3x3_eligibility(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups):
            return super().forward(input)  # geometry or this call is out of scope
        backend = select_backend(input.dtype, input.shape[1], self.weight.shape[0], input.shape[2], input.shape[3])
        if backend == "kernel":
            packed, bias = prepacked_conv3x3(self)
            return conv3x3_prepacked(input, packed, bias)
        if backend == "miopen":
            return conv2d_miopen(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return super().forward(input)
