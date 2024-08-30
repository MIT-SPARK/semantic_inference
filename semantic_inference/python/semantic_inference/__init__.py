"""Module handling open-set segmentation."""

from semantic_inference.config import *
from semantic_inference.misc import Logger

import pathlib


def root_path():
    """Get root path of package."""
    return pathlib.Path(__file__).absolute().parent
