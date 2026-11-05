"""3x3 stride-1 conv2d on gfx11 (bf16/fp16, NCHW) with a prepacked weight.

The kernel is an implicit GEMM: `M = C_out`, `N = H * W` (NCHW pixel order) and `K = 9 * C_in`
with the contraction ordered `k = group * 144 + tap * 16 + (c % 16)` so that the nine taps of one
16-channel group are one contiguous k range. The weight is prepacked once per weight version into
that order. Activations go in as they are (a halo tile per 16-channel group is staged in LDS, so each
input element is read once per block rather than once per tap).

`conv2d` mirrors `torch.nn.functional.conv2d` and silently falls back to it for anything the
kernel does not cover, so it is safe to use as a drop-in.

fp16 3x3 stride-1 is delegated to torch (MIOpen) on purpose: Winograd does 4/9 of the multiplies and
gfx11 has the packed fp16 VALU instructions that make its transforms cheap, which a direct
convolution cannot match. bf16 stays on this kernel, because MIOpen has no bf16 Winograd.
`select_backend` is the single place that decides.
"""

import contextlib
import weakref

import torch
import torch.nn.functional as F
from torch.library import custom_op

from .tiles import CONV_PREFETCH_MIN_C_IN, CONV_TILES, MIOPEN_FP16_MAX_INPUT_ELEMS, conv_block_sizes

# `(block_m, block_n) -> tile` for the one-row instantiations, which is the vocabulary the selector below
# reasons in. The two-row tiles are the same box with a different `rows`.
_TILES_ONE_ROW = {(conv_block_sizes(tile)[0], conv_block_sizes(tile)[1]): tile for tile in CONV_TILES if tile[4] == 1}
_TILES_TWO_ROWS = {(conv_block_sizes(tile)[0], conv_block_sizes(tile)[1]): tile for tile in CONV_TILES if tile[4] == 2}
_BLOCK_MS = tuple(sorted({block_m for block_m, _ in _TILES_ONE_ROW}, reverse=True))
_BLOCK_NS = tuple(sorted({block_n for _, block_n in _TILES_ONE_ROW}, reverse=True))


def select_conv3x3_config(c_out: int, width: int) -> tuple[int, int, int, int, int] | None:
    """`(warps_m, warps_n, repeat_m, repeat_n, rows)` for one convolution, or None if no tile fits.

    The largest `block_m` dividing `c_out` wins, then the largest `block_n` dividing `width`, else the
    largest `block_n` that merely fits - the last column block then overlaps its neighbour and recomputes
    those columns with the same reduction order, so the duplicate stores are identical.

    `rows` is how many output rows one block computes. Two rows stage a 4-row halo for 2 output rows,
    which drops the input re-read from about three times to about twice per output row and halves the
    per-block staging and epilogue work, at the cost of doubling the accumulators. That only fits where
    the single-row tile is narrow in `C_out` (a 128-wide image, or a `C_out` that no 128-wide tile
    divides), which is exactly where it is selected.
    """
    for block_m in _BLOCK_MS:
        if c_out % block_m:
            continue
        block_n = next((bn for bn in _BLOCK_NS if width % bn == 0), 0)
        if not block_n:
            block_n = next((bn for bn in _BLOCK_NS if bn <= width), 0)
            if not block_n:
                return None
        if block_n == 128 and c_out % 128 == 0:
            return _TILES_TWO_ROWS[(128, 128)]
        if block_n == 256 and c_out % 128 and c_out % 64 == 0:
            return _TILES_TWO_ROWS[(64, 256)]
        return _TILES_ONE_ROW[(block_m, block_n)]
    return None


def conv3x3_prefetch(tile: tuple[int, int, int, int, int], c_in: int) -> bool:
    """Whether to prefetch the next channel group into registers.

    The prefetch hides the next group's global load behind this group's compute, so it needs enough groups
    and a wide enough N panel to pay. Measured per cell (`bench/benchmark_conv3x3.py --variants both`):
    +3.5 to +7.9% on every `block_n = 256` tile with `C_in >= 128`, +6.1% on the 128x128 two-row tile, and
    -8 to -9% on the 64x256 two-row tile, which is where the doubled accumulator set leaves no room to
    preload. The variant is instantiated for every tile, so this is a pure runtime choice.
    """
    block_m, block_n, rows = conv_block_sizes(tile)
    if c_in < CONV_PREFETCH_MIN_C_IN or block_n < 128:
        return False
    return not (rows == 2 and block_m == 64)


def select_backend(dtype: torch.dtype, c_in: int, c_out: int, height: int, width: int) -> str:
    """Which implementation should run this 3x3 stride-1 convolution.

    `"kernel"` for `conv3x3_prepacked`, `"miopen"` for the delegated fp16 path (`F.conv2d` with
    cuDNN/MIOpen forced on), `"torch"` when the kernel has no tile for this shape.

    fp16 is delegated while the input is small enough to stay cache-resident, because MIOpen's fp16
    Winograd does 4/9 of the multiplies and gfx11 has the packed fp16 VALU instructions that make its
    transforms cheap, which a direct convolution cannot match. Past that footprint the activation
    stream goes cache-cold and the kernel takes over. bf16 is never delegated: MIOpen has no bf16
    Winograd, and its best bf16 path is slower than this kernel.
    """
    if dtype == torch.float16 and c_in * height * width <= MIOPEN_FP16_MAX_INPUT_ELEMS:
        return "miopen"
    if dtype in (torch.bfloat16, torch.float16) and select_conv3x3_config(c_out, width) is not None:
        return "kernel"
    return "torch"


@contextlib.contextmanager
def _miopen_flags():
    """Force `cudnn.enabled = True` and `cudnn.benchmark = False` for one call, then restore.

    `torch.backends.cudnn.flags()` would do the same, but it also re-sets `benchmark_limit`,
    which MIOpen does not support and warns about on every call.
    """
    enabled = torch.backends.cudnn.enabled
    benchmark = torch.backends.cudnn.benchmark
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    try:
        yield
    finally:
        torch.backends.cudnn.enabled = enabled
        torch.backends.cudnn.benchmark = benchmark


def _conv2d_miopen(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """`F.conv2d` with cuDNN/MIOpen forced on, for the delegated fp16 path.

    ComfyUI sets `torch.backends.cudnn.enabled = False` on AMD, and under that flag the delegated
    call would silently run the im2col path, which is slower than this kernel. Forcing the flag per
    call keeps the delegation honest without changing the caller's global state.
    """
    # benchmark=False pins MIOpen to its heuristic find (no exhaustive search, no autotune), which
    # is the point of delegating: the solver is picked once by MIOpen's predictor, per shape.
    with _miopen_flags():
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


def prepack_conv2d_weight_3x3(weight: torch.Tensor) -> torch.Tensor:
    """`[C_out, C_in, 3, 3]` -> `[ceil(C_in/16) * 18, C_out, 8]` in the kernel's k order.

    `C_in` is zero padded to a multiple of 16, so any channel count works. The result is a pure
    permutation of the weight (plus the padding), so packing is cheap and lossless.
    """
    if weight.ndim != 4 or tuple(weight.shape[2:]) != (3, 3):
        raise ValueError(f"expected a [C_out, C_in, 3, 3] weight, got {tuple(weight.shape)}")
    c_out, c_in = weight.shape[0], weight.shape[1]
    groups = (c_in + 15) // 16
    w = weight.detach().reshape(c_out, c_in, 9)
    if groups * 16 != c_in:
        padded = w.new_zeros((c_out, groups * 16, 9))
        padded[:, :c_in] = w
        w = padded
    # k order: (group, tap, channel % 16)
    w = w.reshape(c_out, groups, 16, 9).permute(1, 3, 2, 0).reshape(groups * 9 * 16, c_out)
    return w.view(groups * 18, 8, c_out).permute(0, 2, 1).contiguous()


def _launch(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    bias: torch.Tensor | None,
    out: torch.Tensor,
    tile: tuple[int, int, int, int, int],
    prefetch: bool,
) -> None:
    torch.ops.feather_ops.conv2d_3x3.default(x, w_packed, bias, out, *tile, prefetch)


@custom_op("feather_ops::conv3x3_auto", mutates_args=("out",))
def _conv3x3_auto(x: torch.Tensor, w_packed: torch.Tensor, bias: torch.Tensor | None, out: torch.Tensor) -> None:
    """The tile and prefetch heuristics behind an opaque boundary, so a compiled caller sees one node."""
    tile = select_conv3x3_config(w_packed.shape[1], x.shape[3])
    if tile is None:
        raise RuntimeError(f"no kernel tile for C_out={w_packed.shape[1]} W={x.shape[3]}")
    _launch(x, w_packed, bias, out, tile, conv3x3_prefetch(tile, x.shape[1]))


def conv3x3_prepacked(
    x: torch.Tensor,
    w_packed: torch.Tensor,
    bias: torch.Tensor | None = None,
    tile: tuple[int, int, int, int, int] | None = None,
    prefetch: bool | None = None,
) -> torch.Tensor:
    """3x3, stride 1, padding 1 conv over a prepacked weight. No fall back, no checking of extras.

    `C_in` must be a multiple of 8. It need not be a multiple of 16 (the prepack zero pads). `tile` and
    `prefetch` default to the heuristics. They exist so a benchmark or a test can pin them.
    """
    out = torch.empty((x.shape[0], w_packed.shape[1], x.shape[2], x.shape[3]), device=x.device, dtype=x.dtype)
    if tile is None:
        _conv3x3_auto(x, w_packed, bias, out)
    else:
        _launch(x, w_packed, bias, out, tile, conv3x3_prefetch(tile, x.shape[1]) if prefetch is None else prefetch)
    return out


def weight_function_is_absent(module: torch.nn.Module) -> bool:
    """ComfyUI compatibility: a patched weight (LoRA / low-VRAM) must not use the prepacked copy."""
    return getattr(module, "weight_function", None) is None and getattr(module, "bias_function", None) is None


def _can_use_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
    groups: int,
) -> bool:
    if groups != 1 or x.ndim != 4:
        return False
    if x.dtype not in (torch.bfloat16, torch.float16) or weight.dtype != x.dtype:
        return False
    if not (x.is_cuda and weight.is_cuda and x.device == weight.device):
        return False
    if tuple(stride) != (1, 1) or tuple(padding) != (1, 1) or tuple(dilation) != (1, 1):
        return False
    if tuple(weight.shape[2:]) != (3, 3):
        return False
    if weight.shape[1] != x.shape[1] or x.shape[1] % 8:
        return False
    if not (x.is_contiguous() and weight.is_contiguous()):
        return False
    if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
        return False
    return not (bias is not None and (bias.dtype != x.dtype or bias.numel() != weight.shape[0]))


def _triple(value) -> tuple[int, ...]:
    if isinstance(value, int):
        return (value, value)
    return tuple(int(v) for v in value)


def weight_cache_key(weight: torch.Tensor) -> tuple:
    """Identity of a weight version for the prepack cache.

    Inference tensors (`torch.inference_mode`) do not carry a version counter, and ComfyUI runs
    node bodies under inference mode, so fall back to the storage identity alone there.
    """
    if torch.is_inference_mode_enabled():
        return (weight.data_ptr(), None, weight.dtype, weight.device)
    version = getattr(weight, "_version", None)
    return (weight.data_ptr(), version, weight.dtype, weight.device)


class _PackedWeightCache:
    """Weight prepack cache for the functional API, keyed by storage identity and version.

    The key alone is not enough: a freed tensor's address can be handed to a new tensor, which then
    starts at version 0 with the same shape, so the entry keeps a weak reference and only hits when
    it is the same tensor object (or an inference-mode view of it, which has no version counter).
    """

    def __init__(self, max_entries: int = 128) -> None:
        self._max_entries = max_entries
        self._entries: dict[tuple, tuple[torch.Tensor, weakref.ref]] = {}

    def get(self, weight: torch.Tensor) -> torch.Tensor:
        key = (*weight_cache_key(weight), weight.shape)
        entry = self._entries.get(key)
        if entry is not None:
            packed, owner = entry
            kept = owner()
            if kept is not None and kept.data_ptr() == weight.data_ptr():
                return packed
        if len(self._entries) >= self._max_entries:
            self._entries.pop(next(iter(self._entries)))
        packed = prepack_conv2d_weight_3x3(weight)
        self._entries[key] = (packed, weakref.ref(weight))
        return packed


_PACKED_CACHE = _PackedWeightCache()


def conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride=1,
    padding=0,
    dilation=1,
    groups: int = 1,
) -> torch.Tensor:
    """Drop-in replacement for `torch.nn.functional.conv2d`.

    Runs the HIP kernel for bf16 NCHW 3x3 stride-1 padding-1 convolutions, delegates fp16 to MIOpen's
    Winograd (see `select_backend`), and falls back to `F.conv2d` for everything else, so callers
    never need to check.
    """
    stride_t, padding_t, dilation_t = _triple(stride), _triple(padding), _triple(dilation)
    if _can_use_kernel(input, weight, bias, stride_t, padding_t, dilation_t, groups):
        backend = select_backend(input.dtype, input.shape[1], weight.shape[0], input.shape[2], input.shape[3])
        if backend == "kernel":
            return conv3x3_prepacked(input, _PACKED_CACHE.get(weight), bias)
        if backend == "miopen":
            return _conv2d_miopen(input, weight, bias, stride, padding, dilation, groups)
    return F.conv2d(input, weight, bias, stride, padding, dilation, groups)


class Conv2d(torch.nn.Conv2d):
    """`nn.Conv2d` that keeps a prepacked copy of its weight for the HIP 3x3 kernel (bf16 only).

    `state_dict` is unchanged: the prepack lives in a non-persistent buffer and is rebuilt lazily
    after a weight update, a dtype/device move, or a `load_state_dict`.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer("_packed_weight", torch.empty(0), persistent=False)
        self.register_buffer("_packed_bias", None, persistent=False)
        self._packed_key = None
        self._packed_enabled = self._kernel_applies()

    def _kernel_applies(self) -> bool:
        return (
            self.groups == 1
            and tuple(self.kernel_size) == (3, 3)
            and tuple(self.stride) == (1, 1)
            and tuple(self.padding) == (1, 1)
            and tuple(self.dilation) == (1, 1)
            and self.padding_mode == "zeros"
        )

    def invalidate_packed_weight(self) -> None:
        self._packed_key = None
        self._packed_weight = torch.empty(0, device=self.weight.device, dtype=self.weight.dtype)
        self._packed_bias = None

    def _load_from_state_dict(self, *args, **kwargs) -> None:
        super()._load_from_state_dict(*args, **kwargs)
        self.invalidate_packed_weight()

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse)
        self.invalidate_packed_weight()
        return result

    def pack_weight(self) -> None:
        if not self._packed_enabled or self.weight.dtype not in (torch.bfloat16, torch.float16):
            return
        self._packed_weight = prepack_conv2d_weight_3x3(self.weight)
        if self.bias is None:
            self._packed_bias = None
        elif self.bias.dtype == self.weight.dtype and self.bias.is_contiguous():
            self._packed_bias = self.bias
        else:
            self._packed_bias = self.bias.detach().to(self.weight.dtype).contiguous()
        self._packed_key = weight_cache_key(self.weight)

    def _in_scope(self, input: torch.Tensor) -> bool:
        """Whether this call is a 3x3 stride-1 NCHW conv the fast paths can take at all."""
        return self._packed_enabled and weight_function_is_absent(self) and _can_use_kernel(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._in_scope(input):
            backend = select_backend(input.dtype, input.shape[1], self.weight.shape[0], input.shape[2], input.shape[3])
            if backend == "kernel":
                if self._packed_key != weight_cache_key(self.weight) or self._packed_weight.numel() == 0:
                    self.pack_weight()
                return conv3x3_prepacked(input, self._packed_weight, self._packed_bias)
            if backend == "miopen":
                return _conv2d_miopen(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return F.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
