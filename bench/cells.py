"""The shared benchmark shape grid: one cell table for every GEMM kernel in this repo.

A cell is `(M, N_w, K_w)`: `M` activation rows, and a weight of `N_w` outputs over `K_w` inputs, i.e. the same
orientation as the prepacked weights (fp8 `(K/16, N, 16)`, int8 `[N, K]` micro-tiles). Both kernels run these
cells: the fp8 one forwards only, the int8 one in both directions (the backward contracts `N_w`, so its output
is `K_w`).

The grid is a deliberate sample, not a cross product. It has three jobs:
- Generalize to the workload. Every model family in `input_shapes_tn.md` contributes at least two of its
  real `(out, in)` pairs, at one or two of its real activation-row bands, so a tile rule fitted here lands in
  the right class for a model whose exact cell is not listed. The pairs below are copied from that document's
  "Shape Families" table. A few of its degenerate pairs are skipped (`in` or `out` of 8 or 12 does not satisfy
  the 16-element granularity either kernel needs).
- Cover the image-size ladder. The workload only runs 1024^2 and 1536^2 in anger. `ladder-*` adds 512^2,
  768^2, 1280^2 and 2048^2 on the three boxes whose classes differ (a wide 6144x6144 weight, a 1536-wide
  output where the fp8 and int8 kernels disagree most, and a 1024x5120 box whose grid is the smallest). With a
  16x VAE those resolutions are 1024, 2304, 4096, 6400, 9216 and 16384 tokens, and the packed models add their
  16 text rows, hence 1040 / 2320 / 4112 / 6416 / 9232 / 16400.
- Hit the regimes the plans say are special. Each group's cells are annotated with the regime they exist
  for: `M <= 256` launch-fill, 512-row text contexts, 5251/25343 ragged rows against their aligned
  neighbours, outputs past 16384, contractions past 8192, and the two weight-streaming cells (248320- and
  96768-wide) where the kernel is at the DRAM roof.

Budget: about 100 cells. The previous table was 139 (fp8) and 68 (int8) from two different lists with 52 cells
in common, so the shared grid is smaller than either and both benches got faster.
"""

# 16x-VAE token counts for the image sizes the ladder covers, plus the 16 packed text rows.
M_512 = 1040
M_768 = 2320
M_1024 = 4112
M_1280 = 6416
M_1536 = 9232
M_2048 = 16400

# group -> [(M, N_w, K_w)]. The group name is what the benchmark prints, so it doubles as the model label.
GROUPS: dict[str, list[tuple[int, int, int]]] = {
    # The resolution ladder on three shape classes. 1024^2 and 1536^2 are also in the model groups below.
    # These rows exist so a resolution change moves through every class with the same M.
    "ladder-wide": [(M_512, 6144, 6144), (M_768, 6144, 6144), (M_1024, 6144, 6144), (M_1280, 6144, 6144), (M_1536, 6144, 6144), (M_2048, 6144, 6144)],
    "ladder-narrow": [(M_512, 1536, 6144), (M_768, 1536, 6144), (M_1024, 1536, 6144), (M_1280, 1536, 6144), (M_1536, 1536, 6144), (M_2048, 1536, 6144)],
    "ladder-small": [(M_512, 1024, 5120), (M_768, 1024, 5120), (M_1024, 1024, 5120), (M_1536, 1024, 5120), (M_2048, 1024, 5120)],
    # SDXL: a legacy fp16 model, kept for its 8x-VAE geometry (the 2304-row band is 768^2 there) and for the
    # 1280-wide output at a short contraction, which no other family has.
    "sdxl": [(1024, 1280, 1280), (1024, 10240, 1280), (2304, 1280, 1280), (2304, 10240, 1280), (4096, 1280, 5120), (9216, 1280, 5120), (4096, 5120, 1280)],
    # Anima: the 512-row and 4096/9216-row bands, wide and narrow outputs, and a 1024-wide output over a long
    # contraction (its weakest real cell).
    "anima": [(512, 1024, 1024), (512, 2048, 1024), (4096, 2048, 2048), (4096, 8192, 2048), (4096, 2048, 8192), (9216, 2048, 2048), (9216, 8192, 2048), (4096, 1024, 4096)],
    # Qwen-Image 1: 16-row text band, 4096/9216 image bands, and the 18432-wide MLP.
    "qwen21": [(4096, 3072, 3072), (4096, 12288, 3072), (4096, 3072, 12288), (9216, 18432, 3072)],
    # Qwen-Image 2.1: the packed 4112/9232 rows, the 2-row modulation band, and the 24576-wide gate_up.
    "qwen2_1": [(2, 4096, 16384), (16, 4096, 4096), (4112, 4096, 4096), (4112, 4096, 12288), (4112, 24576, 4096), (9232, 16384, 4096), (9232, 4096, 4096)],
    # Z-Image: the padded 4128/9248 bands, the 32-row refiner band, and the 3840-wide projections.
    "zimage": [(32, 3840, 2560), (4128, 3840, 3840), (4128, 11520, 3840), (4128, 3840, 10240), (9248, 10240, 3840), (9248, 3840, 3840)],
    # Krea 2: the 192-row prefix band, the 16384-wide output, the long-contraction MLP and the 6912 cell. Its
    # 4112/9232 rows on 6144x6144 and 1536x6144 are the `ladder-*` cells, so they are not repeated here.
    "krea2": [(16, 6144, 6144), (192, 1536, 6144), (4112, 6144, 16384), (4112, 16384, 6144), (4112, 2560, 6912)],
    # Ideogram 4: a 4608-wide model with a 53248-long contraction (the widest weight in the corpus).
    "ideogram4": [(4096, 4608, 4608), (4096, 4608, 12288), (4096, 4608, 53248), (9232, 12288, 4608), (9232, 18432, 512)],
    # Klein 9B: 512-row text context plus the 4096/4608/9216 image bands and the 36864-wide output.
    "klein9b": [(512, 4096, 4096), (4096, 4096, 4096), (4096, 4096, 12288), (4608, 36864, 4096), (4608, 4096, 16384), (9216, 12288, 4096), (9728, 4096, 4096)],
    # Wan: the two video bands, and a 13824-wide output over a 5120 contraction.
    "wan": [(512, 5120, 5120), (512, 13824, 5120), (12032, 5120, 5120), (12032, 5120, 13824), (12032, 30720, 5120)],
    # LTX 2.3: 1024/1536/8960 bands on a 4096-wide model, including the 16384-wide output.
    "ltx2_3": [(1024, 4096, 4096), (1536, 4096, 4096), (8960, 4096, 16384), (8960, 8192, 4096)],
    # H3: the tall 5376-row video band and its 25344-row long-video counterpart, the 7168/14336 MLP, the
    # 21504/28672-wide outputs, and the ragged rows against the aligned 5376.
    "h3": [
        (16, 5376, 5376),
        (512, 5376, 5120),
        (5251, 21504, 5376),
        (5376, 21504, 5376),
        (5376, 5376, 7168),
        (5376, 28672, 5376),
        (5376, 5376, 14336),
        (25343, 21504, 5376),
        (25344, 21504, 5376),
    ],
    # MOSS-Image: the fused 9216-wide qkv, the 3072-wide body, and the 18432-wide shared modulation.
    "moss": [(4112, 9216, 3072), (4112, 3072, 3072), (4112, 16384, 3072), (9232, 3072, 8192), (9232, 18432, 3072)],
    # Small-M tiers the plans fitted separately: short prompts (32/128), the 192/256 launch-fill band, and the
    # two text-encoder widths (1152 and 4304) that the 64-wide output rule exists for.
    "short-prompt": [(32, 25600, 5120), (32, 5120, 25600), (128, 1024, 5120), (128, 8192, 5120)],
    # The 16-row text band, shared by every model whose text embedder runs there (Qwen-Image 1 and MOSS-Image
    # both hit this exact box), kept once so the bench does not measure it twice.
    "text-16": [(16, 3072, 3072)],
    "pad-256": [(192, 2560, 2560), (256, 2560, 2560), (192, 12288, 4096), (256, 12288, 4096), (256, 4096, 4096)],
    "text-1152": [(512, 1152, 1152), (2048, 1152, 1152), (2048, 1152, 4304), (2048, 4304, 1152), (4096, 3456, 1152)],
    # Weight streaming at the DRAM roof: the 32B text encoder's vocabulary projection and H3's adaln_proj.
    "weight-stream": [(512, 248320, 4096), (1, 96768, 2688)],
}


def cells(groups: set[str] | None = None) -> list[tuple[str, int, int, int]]:
    """`[(group, M, N_w, K_w)]` for the grid, or the subset in `groups`.

    A cell belongs to exactly one group: the `ladder-*` groups own the resolutions they define, so a model
    group must not repeat them (the assert below is what keeps the bench honest, since a duplicated cell is
    measured twice and counted twice).
    """
    selected = [(group, m, n, k) for group, group_cells in GROUPS.items() if groups is None or group in groups for m, n, k in group_cells]
    shapes = [(m, n, k) for _, m, n, k in selected]
    assert len(set(shapes)) == len(shapes), f"the grid repeats a cell: {[shape for shape in set(shapes) if shapes.count(shape) > 1]}"
    return selected


def group_names() -> list[str]:
    return list(GROUPS)
