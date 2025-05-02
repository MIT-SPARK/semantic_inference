import pathlib
import shutil
import subprocess

from setuptools import setup
from torch.utils import cpp_extension

package_path = pathlib.Path(__file__).absolute().parent
mask2former_path = package_path / "third_party" / "Mask2Former" / "mask2former"
ops_path = mask2former_path / "modeling" / "pixel_decoder" / "ops"

if not mask2former_path.exists():
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Missing git and submodule is not initialized")

    subprocess.run([git, "submodule", "--init", "update"])

files = [
    str(x) for x in list(ops_path.glob("**/*.cpp")) + list(ops_path.glob("**/*.cu"))
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
