"""Catkin setup for semantic_inference_ros."""

from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=["semantic_inference_ros"], package_dir={"": "src"}
)
setup(**setup_args)
