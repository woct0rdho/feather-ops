# FeatherOps

## How it works

The idea is from https://github.com/SuriyaaMM/feather . On GPUs without native fp8, when doing matmul, we can load fp8 data to LDS/VGPR then upcast to fp16 before WMMA, rather than upcast the whole matrix to fp16 then load it to LDS/VGPR. This reduces LDS usage and improves occupancy. It also takes less instructions in LDS -> VGPR load, even taking account of fp8e5m2 -> fp16 upcast, which improves compute-load overlap in the K-loop.

On Nvidia consumer Pascal (GTX 10xx, not P100, and not Volta), Turing, and Ampere GPUs, as well as AMD RDNA2 GPUs, int8 ConvRot models are preferable to fp8 models, which actually reduce the compute time with faster int8 matmul than fp16.

It's a pity that RDNA3/3.5 GPUs do not have faster int8 matmul than fp16. int8 ConvRot models can still be faster than bf16 models because of the saved LDS/VGPR usage and bandwidth. FeatherOps is an alternative way to speedup fp8e5m2 models.

## Implementation

`csrc/hip_kernel.cu` is the mixed precision kernel. Other kernels are for experiments.

The kernel is written in HIP, with intrinsics and asm when needed, without abstraction levels like CK or Triton.

The kernel computes fp16 @ fp8e5m2 -> fp16/bf16. fp8 @ fp8 seems not achieving further speedup. We choose fp8e5m2 rather than fp8e4m3, and fp16 rather than bf16 for input, because it's extremely fast to upcast fp8e5m2 -> fp16. We use fp32 accumulator in the wmma, and downcast to fp16/bf16 for output depending on the model's compute dtype.

The kernel requires the inputs to be aligned with the M/N/K block sizes, and there are no branches to handle OOB cases. This is satisfied in most models.

We prepack the B matrix into `(K/16, N, 16)` layout to enable fast 128-bit VRAM -> LDS load, and ensure that threads with adjacent N load adjacent 128-bit elements. Note that the B matrix (weight) is in `(N, K)` rather than `(K, N)` layout in usual ComfyUI workloads.

On RDNA3, there are no ways like cp.async to explicitly control compute-load overlap, so for now we can only write a serial pipeline and hope it work well with the hardware scheduler. It seems setting `s_setprio 1` in the LDS -> VGPR load stages except the first stage improves the overall speed.

Split-K is implemented in the `split-k` branch but for now there is no notable speedup.

The kernel is tested on Strix Halo, and it should also work with RDNA3 GPUs.

For use in ComfyUI, see https://github.com/woct0rdho/ComfyUI-FeatherOps . Docs are also maintained there.
