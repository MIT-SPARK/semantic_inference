import pathlib
import shutil
import subprocess

from setuptools import setup

package_path = pathlib.Path(__file__).absolute().parent
mask2former_path = package_path / "third_party" / "Mask2Former" / "mask2former"

if not mask2former_path.exists():
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Missing git and submodule is not initialized")

    subprocess.run([git, "submodule", "--init", "update"])

setup()
