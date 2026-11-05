"""Backend flags the device kernels are compiled with, shared by `setup.py` and `tools/check_lds.py`.

They live here rather than in `setup.py` so the LDS audit compiles exactly what the build ships. A tool that
compiles with different flags can report an allocation the shipped binary does not have (or miss one it
does).
"""

DEVICE_FLAGS = [
    "-mllvm",
    "-disable-promote-alloca-to-lds",
]
