# FeatherOps

This package provides fast quantized matmul kernels for RDNA3/3.5 GPUs. The kernels are tested on Strix Halo, and they should also work with RDNA3 GPUs.

For use in ComfyUI, see https://github.com/woct0rdho/ComfyUI-FeatherOps . Docs are also maintained there.

## fp16-fp8 matmul

The name 'feather' is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can load fp8 data to LDS/VGPR then upcast to fp16 before WMMA, rather than upcast the whole matrix to fp16 then load it to LDS/VGPR. This reduces LDS usage and improves occupancy. It also takes less instructions in LDS -> VGPR load, even taking account of fp8e5m2 -> fp16 upcast, which improves compute-load overlap in the K-loop.

`csrc/hip_kernel.cu` is the fp16-fp8 kernel. The kernel is written in HIP, with intrinsics and asm when needed, without abstraction levels like CK or Triton.

The kernel computes fp16 @ fp8e5m2 -> fp16/bf16. fp8 @ fp8 seems not achieving further speedup. We choose fp8e5m2 rather than fp8e4m3, and fp16 rather than bf16 for input, because it's extremely fast to upcast fp8e5m2 -> fp16. We use fp32 accumulator in the WMMA, and downcast to fp16/bf16 for output depending on the model's compute dtype.

The kernel requires the inputs to be aligned with the M/N/K block sizes, and there are no branches to handle OOB cases. This is satisfied in most models.

We prepack the B matrix into `(K/16, N, 16)` layout to enable fast 128-bit VRAM -> LDS load, and ensure that threads with adjacent N load adjacent 128-bit elements.

On RDNA3, there is no way like cp.async to explicitly control compute-load overlap, so we rely on the hardware scheduler. It seems setting `s_setprio 1` in the LDS -> VGPR load stages except the first stage improves the overall speed.

Split-K is implemented in the `split-k` branch but for now there is no notable speedup.

## int8 ConvRot prepacked matmul

`csrc/hip_kernel_int8.cu` is the int8 kernel. It computes int8 @ int8 -> fp16/bf16, with a per-row activation scale and a per-output-channel weight scale, as in the ComfyUI int8 tensorwise format. The Hadamard rotation is fused into the activation producer.

We prepack the B matrix into `(K/4, N/4, 4, 4)` layout to enable fast 128-bit VRAM -> LDS load for both fwd and bwd.
