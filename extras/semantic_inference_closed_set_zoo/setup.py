import pathlib
import shutil
import subprocess

import torch
from setuptools import setup
from torch.utils import cpp_extension

package_path = pathlib.Path(__file__).absolute().parent
mask2former_path = package_path / "third_party" / "Mask2Former" / "mask2former"
ops_path = mask2former_path / "modeling" / "pixel_decoder" / "ops"

if not mask2former_path.exists():
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Missing git and submodule is not initialized")

    subprocess.run([git, "submodule", "update", "--init"])

if not torch.cuda.is_available():
    raise RuntimeError("torch.cuda.available() is required!")

files = [
    "third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/vision.cpp",
    "third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cpu/ms_deform_attn_cpu.cpp",
    "third_party/Mask2Former/mask2former/modeling/pixel_decoder/ops/src/cuda/ms_deform_attn_cuda.cu",
]

setup(
    ext_modules=[
        cpp_extension.CUDAExtension(
            "MultiScaleDeformableAttention",
            files,
            include_dirs=[str(ops_path / "src")],
            extra_compile_args={
                "cxx": ["-Wno-deprecated-declarations"],
                "nvcc": [
                    "-Wno-deprecated-declarations",
                    "-DCUDA_HAS_FP16=1",
                    "-D__CUDA_NO_HALF_OPERATORS__",
                    "-D__CUDA_NO_HALF_CONVERSIONS__",
                    "-D__CUDA_NO_HALF2_OPERATORS__",
                ],
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
