"""The generated instantiation header, and the tiles the heuristics can return.

The tables in `feather_ops/tiles.py` are the source the kernels are generated from, so these two checks
are the only place the heuristics and the kernel can disagree.
"""

import pathlib
import subprocess
import sys

from feather_ops import conv2d, int8_convrot, scaled_mm
from feather_ops.tiles import CONV_TILES, INT8_TILES, SCALED_MM_TILES, conv_block_sizes

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULES = {scaled_mm: SCALED_MM_TILES, int8_convrot: INT8_TILES, conv2d: CONV_TILES}


def test_generated_header_is_current():
    result = subprocess.run([sys.executable, str(ROOT / "tools/gen_tiles.py"), "--check"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"{result.stdout}{result.stderr}"


def test_heuristic_tiles_are_instantiated():
    for module, table in MODULES.items():
        for name in dir(module):
            if name.startswith("_TILE_"):
                assert getattr(module, name) in table, f"{module.__name__}.{name} is not in the tile table"


def test_no_duplicate_tiles():
    for table in MODULES.values():
        assert len(set(table)) == len(table)
    # The conv selector maps (block_m, block_n) to an instantiation, so those boxes must be unique.
    boxes = [conv_block_sizes(tile) for tile in CONV_TILES]
    assert len(set(boxes)) == len(boxes)
