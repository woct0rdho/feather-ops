import importlib.util
import os
import shlex
import shutil
import sys
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils import cpp_extension

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import tools.gen_tiles as _gen_tiles

CSRC = ROOT / "csrc"
SOURCES = [
    "csrc/extension.cpp",
    "csrc/hip_conv2d_3x3.cu",
    "csrc/hip_kernel.cu",
    "csrc/hip_kernel_fp16.cu",
    "csrc/hip_kernel_int8.cu",
    "csrc/hipblaslt_kernel_fp16.cu",
]
# Absolute paths: setuptools feeds these to ninja as prerequisites, and a relative prerequisite is not
# matched against the compiled object's dependency list.
HEADER_DEPENDENCIES = [str(path) for pattern in ("*.h", "*.cuh") for path in sorted(CSRC.rglob(pattern))]


def _enable_ccache() -> None:
    ccache = shutil.which("ccache")
    if ccache is None:
        return

    rocm_home = cpp_extension.ROCM_HOME
    hipcc = shutil.which("hipcc")
    if hipcc is None and rocm_home is not None:
        candidate = Path(rocm_home) / "bin" / "hipcc"
        if candidate.is_file():
            hipcc = str(candidate)
    if hipcc is not None:
        os.environ.setdefault(
            "PYTORCH_NVCC",
            f"{shlex.quote(ccache)} {shlex.quote(hipcc)}",
        )

    if "CXX" not in os.environ:
        for candidate in (Path("/usr/lib/ccache/c++"), Path("/usr/lib64/ccache/c++")):
            if candidate.is_file():
                os.environ["CXX"] = str(candidate)
                break


def _rocm_devel_root() -> Path:
    if importlib.util.find_spec("rocm_sdk._devel") is not None:
        from rocm_sdk._devel import get_devel_root

        return Path(get_devel_root())

    if cpp_extension.ROCM_HOME is None:
        raise RuntimeError("Cannot locate the ROCm development tree. Install rocm-sdk-devel or set ROCM_HOME/ROCM_PATH.")
    return Path(cpp_extension.ROCM_HOME)


if not cpp_extension.IS_HIP_EXTENSION:
    raise RuntimeError("feather-ops requires a ROCm-enabled PyTorch build.")

_enable_ccache()

rocm_devel_root = _rocm_devel_root()
rocm_include_dir = rocm_devel_root / "include"
include_dirs = [str(CSRC), str(rocm_include_dir)]
rocm_lib_dirs = [str(rocm_devel_root / "lib")]

# The hipBLASLt kernels are for experiments and not exported in the public API of feather-ops.
# If you only need to use the public API, then it's optional to build the hipBLASLt kernels.
libraries = []
extra_link_args = []
if os.name == "nt":
    extra_link_args.append("libhipblaslt.dll.a")
else:
    libraries.append("hipblaslt")
    extra_link_args.extend(f"-Wl,-rpath,{lib_dir}" for lib_dir in rocm_lib_dirs)

stable_defines = [
    "-DTORCH_TARGET_VERSION=0x020A000000000000",
    "-DTORCH_STABLE_ONLY",
]
common_flags = [
    "-O3",
    "-Wall",
    "-Wextra",
    "-Werror",
    "-Wno-deprecated-copy-with-user-provided-copy",
    "-Wno-ignored-qualifiers",
    "-Wno-shift-count-overflow",
    "-Wno-unused-function",
    "-Wno-unused-parameter",
    *stable_defines,
]
if os.name == "nt":
    # PyTorch's stable headers use deprecated CRT APIs with recent MSVC.
    common_flags.append("-D_CRT_SECURE_NO_WARNINGS")

hip_flags = [
    *common_flags,
    "-U__HIP_NO_HALF_CONVERSIONS__",
    "-U__HIP_NO_HALF_OPERATORS__",
    "-U__HIP_NO_HALF2_OPERATORS__",
    "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
    "-U__HIP_NO_BFLOAT16_OPERATORS__",
    "-U__HIP_NO_BFLOAT162_OPERATORS__",
]
if os.name == "nt":
    # ROCm's injected HIP wrapper pulls in MSVC <cmath> too early on Windows.
    hip_flags.append("-nohipwrapperinc")

# The tile tables in feather_ops/tiles.py are what the kernels are instantiated from, so the header is
# regenerated on every build. `python tools/gen_tiles.py --check` (run by tests/test_tiles.py) catches a
# hand-edited header.
_gen_tiles.HEADER.write_text(_gen_tiles.render())

setup(
    # `package_dir` maps the root package to the repository root, which makes the layout
    # "simple" in setuptools' terms: an editable install then extends `sys.path` with a plain
    # path instead of installing an import hook that static analysis tools cannot follow. The
    # mapping must be `.`, because an empty string is taken as the egg base by `egg_info`.
    package_dir={"": "."},
    packages=find_packages(exclude=("bench", "cpp_benchmarks", "profile", "tests")),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="feather_ops._C",
            sources=SOURCES,
            include_dirs=include_dirs,
            library_dirs=rocm_lib_dirs,
            libraries=libraries,
            depends=HEADER_DEPENDENCIES,
            extra_compile_args={"cxx": common_flags, "nvcc": hip_flags},
            extra_link_args=extra_link_args,
            py_limited_api=True,
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
