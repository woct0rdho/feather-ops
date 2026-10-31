import pytest
import torch

from feather_ops.convert import bf16_to_fp16

SHAPES = [
    (1024, 1024),
    (257, 512),
    (1, 4096),
]


@pytest.mark.parametrize("shape", SHAPES)
def test_conversion(shape: tuple[int, int]) -> None:
    torch.manual_seed(0)
    x_bf16 = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    y_fp16_custom = bf16_to_fp16(x_bf16)
    y_fp16_torch = x_bf16.to(torch.float16)

    torch.testing.assert_close(y_fp16_custom, y_fp16_torch, rtol=0, atol=1e-5)
