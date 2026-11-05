"""Relabelling an existing module class, so a kernel can take over its `forward`.

A consumer loads a model with its own machinery and then wants a kernel in the forward of the modules
that can take it. Replacing modules would break the framework (parameter identities, patchers, offload
bookkeeping), so the forward is added by giving the instance a class that puts a mixin in front of the
class it already has. The parameters, buffers, module names and `state_dict` keys are untouched, which
is what lets this happen after a checkpoint is loaded.

What the consumers decide, not this module: which classes may be relabelled (ComfyUI allows only the
exact `comfy.ops` bases, sd-scripts also allows its manual-padding `ChunkedConv2d`), the mixin itself
(cast/offload hooks, patch checks, logging) and how a weight is obtained.
"""

from collections.abc import Callable

from torch import nn

_RELABELLED: dict[tuple[type, type], type] = {}


def relabelled_class(cls: type, mixin: type) -> type:
    """A cached subclass of `cls` that puts `mixin` in front of its `forward`.

    One object per `(cls, mixin)` pair, so `type(module)` is stable across calls, models and deepcopies,
    and every `isinstance` check the framework does still holds.
    """
    key = (cls, mixin)
    cached = _RELABELLED.get(key)
    if cached is None:
        cached = type(f"Feather{cls.__name__}", (mixin, cls), {"__module__": __name__, "__qualname__": cls.__qualname__})
        _RELABELLED[key] = cached
    return cached


def relabel_modules(root: nn.Module, mixin: type, accept: Callable[[nn.Module], bool]) -> int:
    """Relabel every module in `root` that `accept(module)` approves, returning how many changed.

    `accept` carries the whole policy of the caller (geometry, allowed base classes, anything else).
    Modules that already carry `mixin` are skipped, so a second call is a no-op.
    """
    count = 0
    for module in root.modules():
        if isinstance(module, mixin) or not accept(module):
            continue
        module.__class__ = relabelled_class(type(module), mixin)
        count += 1
    return count
